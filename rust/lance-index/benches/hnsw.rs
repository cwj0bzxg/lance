// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of HNSW graph.
//!
//!

use std::{collections::HashSet, sync::Arc, time::Duration};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array, UInt64Array,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::v3::subindex::IvfSubIndex;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};
use rayon::ThreadPoolBuilder;

use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::vector::transform::Transformer;
use lance_index::vector::{
    CENTROID_DIST_COLUMN, PART_ID_COLUMN,
    bq::{
        RQBuildParams, builder::RabitQuantizer, storage::RabitQuantizationStorage,
        transform::RQTransformer,
    },
    flat::storage::FlatFloatStorage,
    graph::VisitedGenerator,
    hnsw::builder::{HNSW, HnswBuildParams, HnswQueryParams},
    pq::{PQBuildParams, ProductQuantizer},
    quantizer::{Quantization, QuantizerStorage},
    sq::{ScalarQuantizer, builder::SQBuildParams},
    storage::{StorageBuilder, VectorStore},
};
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;

fn bench_hnsw(c: &mut Criterion) {
    const DIMENSION: usize = 128;
    const TOTAL: usize = 100_000;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 100;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();
    let vectors = Arc::new(FlatFloatStorage::new(fsl.clone(), DistanceType::L2));

    let query = fsl.value(0);
    c.bench_function(format!("create_hnsw({TOTAL}x{DIMENSION})").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let hnsw = HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()).unwrap();
            let uids: HashSet<u32> = hnsw
                .search_basic(
                    query.clone(),
                    K,
                    &HnswQueryParams {
                        ef: 300,
                        lower_bound: None,
                        upper_bound: None,
                        dist_q_c: 0.0,
                        use_acorn: false,
                    },
                    None,
                    vectors.as_ref(),
                )
                .unwrap()
                .0
                .iter()
                .map(|node| node.id)
                .collect();

            assert_eq!(uids.len(), K);
        })
    });

    let search_build_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let hnsw = search_build_pool
        .install(|| HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()))
        .unwrap();
    c.bench_function(format!("search_hnsw{TOTAL}x{DIMENSION}").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let uids: HashSet<u32> = hnsw
                .search_basic(
                    query.clone(),
                    K,
                    &HnswQueryParams {
                        ef: 300,
                        lower_bound: None,
                        upper_bound: None,
                        dist_q_c: 0.0,
                        use_acorn: false,
                    },
                    None,
                    vectors.as_ref(),
                )
                .unwrap()
                .0
                .iter()
                .map(|node| node.id)
                .collect();

            assert_eq!(uids.len(), K);
        })
    });
}

fn bench_hnsw_load(c: &mut Criterion) {
    const DIMENSION: usize = 128;
    const TOTAL: usize = 100_000;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 100;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();
    let vectors = Arc::new(FlatFloatStorage::new(fsl.clone(), DistanceType::L2));

    let search_build_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let hnsw = search_build_pool
        .install(|| HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()))
        .unwrap();
    let batch = hnsw.to_batch().unwrap();

    // Load cost -- the path #6746 targets. `RecordBatch::clone` is an Arrow
    // refcount bump (what production does anyway: each partition-cache IPC
    // read yields a fresh batch), so it does not mask the load work measured.
    c.bench_function(format!("load_hnsw({TOTAL}x{DIMENSION})").as_str(), |b| {
        b.iter(|| {
            let loaded = HNSW::load(batch.clone()).unwrap();
            assert_eq!(loaded.len(), TOTAL);
        })
    });

    // Search on the Arrow-backed loaded graph -- same TOTAL/DIMENSION/K/ef as
    // the `search_hnsw` bench, so the two are directly comparable and confirm
    // the new backend keeps search latency unchanged (issue #6746).
    let loaded = HNSW::load(batch).unwrap();
    let query = fsl.value(0);
    c.bench_function(
        format!("search_hnsw_loaded{TOTAL}x{DIMENSION}").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let uids: HashSet<u32> = loaded
                    .search_basic(
                        query.clone(),
                        K,
                        &HnswQueryParams {
                            ef: 300,
                            lower_bound: None,
                            upper_bound: None,
                            dist_q_c: 0.0,
                            use_acorn: false,
                        },
                        None,
                        vectors.as_ref(),
                    )
                    .unwrap()
                    .0
                    .iter()
                    .map(|node| node.id)
                    .collect();

                assert_eq!(uids.len(), K);
            })
        },
    );
}

fn bench_hnsw_sq(c: &mut Criterion) {
    const DIMENSION: usize = 128;
    const TOTAL: usize = 100_000;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 100;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();
    let quantizer =
        <ScalarQuantizer as Quantization>::build(&fsl, DistanceType::L2, &SQBuildParams::default())
            .unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Field::new_list_field(DataType::Float32, true).into(),
                DIMENSION as i32,
            ),
            true,
        ),
        ROW_ID_FIELD.clone(),
    ]));
    let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(fsl.clone()), Arc::new(row_ids)]).unwrap();
    let sq_storage = StorageBuilder::new("vector".to_owned(), DistanceType::L2, quantizer, None)
        .unwrap()
        .build(vec![batch])
        .unwrap();
    let vectors = Arc::new(sq_storage);

    let query = fsl.value(0);
    c.bench_function(
        format!("create_hnsw_sq({TOTAL}x{DIMENSION})").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let hnsw =
                    HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()).unwrap();
                let uids: HashSet<u32> = hnsw
                    .search_basic(
                        query.clone(),
                        K,
                        &HnswQueryParams {
                            ef: 300,
                            lower_bound: None,
                            upper_bound: None,
                            dist_q_c: 0.0,
                            use_acorn: false,
                        },
                        None,
                        vectors.as_ref(),
                    )
                    .unwrap()
                    .0
                    .iter()
                    .map(|node| node.id)
                    .collect();

                assert_eq!(uids.len(), K);
            })
        },
    );

    let search_build_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let hnsw = search_build_pool
        .install(|| HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()))
        .unwrap();
    c.bench_function(format!("search_hnsw_sq{TOTAL}x{DIMENSION}").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let uids: HashSet<u32> = hnsw
                .search_basic(
                    query.clone(),
                    K,
                    &HnswQueryParams {
                        ef: 300,
                        lower_bound: None,
                        upper_bound: None,
                        dist_q_c: 0.0,
                        use_acorn: false,
                    },
                    None,
                    vectors.as_ref(),
                )
                .unwrap()
                .0
                .iter()
                .map(|node| node.id)
                .collect();

            assert_eq!(uids.len(), K);
        })
    });
}

fn bench_hnsw_pq(c: &mut Criterion) {
    const DIMENSION: usize = 128;
    const TOTAL: usize = 100_000;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 100;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();
    let quantizer = <ProductQuantizer as Quantization>::build(
        &fsl,
        DistanceType::L2,
        &PQBuildParams::new(16, 8),
    )
    .unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Field::new_list_field(DataType::Float32, true).into(),
                DIMENSION as i32,
            ),
            true,
        ),
        ROW_ID_FIELD.clone(),
    ]));
    let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(fsl.clone()), Arc::new(row_ids)]).unwrap();
    let pq_storage = StorageBuilder::new("vector".to_owned(), DistanceType::L2, quantizer, None)
        .unwrap()
        .build(vec![batch])
        .unwrap();
    let vectors = Arc::new(pq_storage);

    let query = fsl.value(0);
    c.bench_function(
        format!("create_hnsw_pq({TOTAL}x{DIMENSION})").as_str(),
        |b| {
            b.to_async(&rt).iter(|| async {
                let hnsw =
                    HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()).unwrap();
                let uids: HashSet<u32> = hnsw
                    .search_basic(
                        query.clone(),
                        K,
                        &HnswQueryParams {
                            ef: 300,
                            lower_bound: None,
                            upper_bound: None,
                            dist_q_c: 0.0,
                            use_acorn: false,
                        },
                        None,
                        vectors.as_ref(),
                    )
                    .unwrap()
                    .0
                    .iter()
                    .map(|node| node.id)
                    .collect();

                assert_eq!(uids.len(), K);
            })
        },
    );

    let search_build_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let hnsw = search_build_pool
        .install(|| HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()))
        .unwrap();
    c.bench_function(format!("search_hnsw_pq{TOTAL}x{DIMENSION}").as_str(), |b| {
        b.to_async(&rt).iter(|| async {
            let uids: HashSet<u32> = hnsw
                .search_basic(
                    query.clone(),
                    K,
                    &HnswQueryParams {
                        ef: 300,
                        lower_bound: None,
                        upper_bound: None,
                        dist_q_c: 0.0,
                        use_acorn: false,
                    },
                    None,
                    vectors.as_ref(),
                )
                .unwrap()
                .0
                .iter()
                .map(|node| node.id)
                .collect();

            assert_eq!(uids.len(), K);
        })
    });
}

/// `IVF_HNSW_RQ` builds its graph from code-to-code distances and searches
/// with the 1-bit walk plus a multi-bit rerank.
fn bench_hnsw_rq(c: &mut Criterion) {
    const DIMENSION: usize = 128;
    const TOTAL: usize = 100_000;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 100;

    let data = generate_random_array_with_seed::<Float32Type>(TOTAL * DIMENSION, SEED);
    let norms: Vec<f32> = data
        .values()
        .as_chunks::<DIMENSION>()
        .0
        .iter()
        .map(|vector| vector.iter().map(|value| value * value).sum())
        .collect();
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();
    let params = RQBuildParams {
        with_sym_columns: true,
        ..Default::default()
    };
    let quantizer = RabitQuantizer::build(&fsl, DistanceType::L2, &params).unwrap();
    // One partition around a zero centroid: the residuals are the vectors.
    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![0.0; DIMENSION]),
        DIMENSION as i32,
    )
    .unwrap();
    let batch = RecordBatch::try_from_iter(vec![
        ("vector", Arc::new(fsl.clone()) as ArrayRef),
        (
            PART_ID_COLUMN,
            Arc::new(UInt32Array::from(vec![0; TOTAL])) as ArrayRef,
        ),
        (
            CENTROID_DIST_COLUMN,
            Arc::new(Float32Array::from(norms.clone())) as ArrayRef,
        ),
        (
            ROW_ID,
            Arc::new(UInt64Array::from_iter_values(0..TOTAL as u64)) as ArrayRef,
        ),
    ])
    .unwrap();
    let encoded = RQTransformer::new(quantizer.clone(), DistanceType::L2, centroids, "vector")
        .unwrap()
        .transform(&batch)
        .unwrap();
    let storage = RabitQuantizationStorage::try_from_batch(
        encoded,
        &quantizer.metadata(None),
        DistanceType::L2,
        None,
    )
    .unwrap();

    c.bench_function(
        format!("create_hnsw_rq({TOTAL}x{DIMENSION})").as_str(),
        |b| b.iter(|| HNSW::index_vectors(&storage, HnswBuildParams::default()).unwrap()),
    );

    let hnsw = HNSW::index_vectors(&storage, HnswBuildParams::default()).unwrap();
    let query = fsl.value(0);
    let params = HnswQueryParams {
        ef: 300,
        lower_bound: None,
        upper_bound: None,
        dist_q_c: norms[0],
        use_acorn: false,
    };
    c.bench_function(format!("search_hnsw_rq{TOTAL}x{DIMENSION}").as_str(), |b| {
        b.iter(|| {
            let results = hnsw
                .search(
                    query.clone(),
                    K,
                    params,
                    &storage,
                    Arc::new(NoFilter),
                    &NoOpMetricsCollector,
                )
                .unwrap();
            assert_eq!(results.num_rows(), K);
        })
    });
}

/// Measure how the look-ahead prefetch distance affects flat-storage search
/// latency across vector sizes (issue #8275).
///
/// `search_basic` hardcodes its bottom-level prefetch distance, so this bench
/// drives `search_inner` directly to vary it. All distances share one graph
/// and one query set per dimension; only the prefetch distance changes.
/// Queries rotate through 64 stored vectors so the per-iteration working set
/// stays larger than the last-level cache and vector reads miss like they do
/// on a real index.
fn bench_hnsw_prefetch(c: &mut Criterion) {
    const TOTAL: usize = 300_000;
    const SEED: [u8; 32] = [42; 32];
    const K: usize = 100;
    const NUM_QUERIES: usize = 64;

    for dimension in [128usize, 768] {
        let data = generate_random_array_with_seed::<Float32Type>(TOTAL * dimension, SEED);
        let fsl = FixedSizeListArray::try_new_from_values(data, dimension as i32).unwrap();
        let vectors = Arc::new(FlatFloatStorage::new(fsl.clone(), DistanceType::L2));

        let hnsw = HNSW::index_vectors(vectors.as_ref(), HnswBuildParams::default()).unwrap();

        let queries: Vec<_> = (0..NUM_QUERIES)
            .map(|i| fsl.value(i * (TOTAL / NUM_QUERIES)))
            .collect();
        let params = HnswQueryParams {
            ef: 300,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };

        for prefetch_distance in [None, Some(1), Some(2), Some(4), Some(8)] {
            let name = format!(
                "search_hnsw_prefetch({TOTAL}x{dimension},{})",
                prefetch_distance.map_or("off".to_string(), |d| d.to_string())
            );
            let mut visited_generator = VisitedGenerator::new(vectors.len());
            let mut next_query = 0;
            c.bench_function(&name, |b| {
                b.iter(|| {
                    let query = queries[next_query % NUM_QUERIES].clone();
                    next_query += 1;
                    let (result, _) = hnsw
                        .search_inner(
                            query,
                            K,
                            &params,
                            None,
                            &mut visited_generator,
                            vectors.as_ref(),
                            prefetch_distance,
                        )
                        .unwrap();
                    assert_eq!(result.len(), K);
                })
            });
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_hnsw, bench_hnsw_load, bench_hnsw_sq, bench_hnsw_pq, bench_hnsw_rq, bench_hnsw_prefetch);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_hnsw, bench_hnsw_load, bench_hnsw_sq, bench_hnsw_pq, bench_hnsw_rq, bench_hnsw_prefetch);

criterion_main!(benches);
