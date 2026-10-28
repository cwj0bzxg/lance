// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SymRaBitQ: distances between two RaBitQ codes.
//!
//! `IVF_HNSW_RQ` builds its graph without the raw vectors, so the edges are
//! scored between two RaBitQ codes with the symmetric estimator in
//! [`sym_dist`]. The graph walk scores the query against the 1-bit codes
//! only: the query is quantized to 4 bits ([`prepare_residual_warmup_query`])
//! so that AND + popcount replaces the float inner product.
//!
//! "Lib" refers to the reference RaBitQ-Library. These kernels reproduce its
//! layouts and numerics; `testdata/symrabitq` holds its outputs.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::LazyLock;

use lance_linalg::distance::DistanceType;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Bytes per row of sign words: one `u64` per 64 dims, the last one
/// zero-padded.
pub fn sign_word_bytes(dim: usize) -> usize {
    dim.div_ceil(64) * 8
}

/// Regroups the sign bits of row-major `_rabit_codes` (dim `i` at bit `i % 8`
/// of byte `i / 8`) into the sign words the kernels read: one little-endian
/// `u64` per 64 dims with the first dim at bit 63, as Lib packs them.
pub fn sign_words_from_codes(codes: &[u8], code_len: usize) -> Vec<u8> {
    let word_bytes = sign_word_bytes(code_len * 8);
    let mut words = vec![0u8; codes.len() / code_len * word_bytes];
    for (row, row_words) in codes
        .chunks_exact(code_len)
        .zip(words.chunks_exact_mut(word_bytes))
    {
        for (bytes, word) in row.chunks(8).zip(row_words.as_chunks_mut::<8>().0) {
            let mut le = [0u8; 8];
            le[..bytes.len()].copy_from_slice(bytes);
            *word = u64::from_le_bytes(le).reverse_bits().to_le_bytes();
        }
    }
    words
}

/// Per-row factors of the symmetric estimator, stored in the `__sym_*`
/// columns. They are not the asymmetric `__add_factors` / `__scale_factors`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SymFactors {
    /// `‖r‖` of the rotated residual `r`.
    pub rho: f32,
    /// `⟨r, ũ⟩ / (‖r‖ ‖ũ‖)` for the centered code `ũ`.
    pub gamma: f32,
    /// `‖ũ‖`.
    pub unorm: f32,
    /// `⟨r, c⟩ + ‖c‖² / 2` for the rotated centroid `c`; only Dot reads it.
    pub ip_cent: f32,
}

/// Accumulates the [`SymFactors`] of one row while it is quantized.
pub struct SymFactorsBuilder<'a> {
    centroid: &'a [f32],
    centroid_norm_sq: f32,
    ex_bits: u8,
    code_bias: f32,
    rho_sq: f32,
    unorm_sq: f32,
    ip: f32,
    ip_cent: f32,
}

impl<'a> SymFactorsBuilder<'a> {
    pub(crate) fn new(centroid: &'a [f32], centroid_norm_sq: f32, num_bits: u8) -> Self {
        Self {
            centroid,
            centroid_norm_sq,
            ex_bits: num_bits - 1,
            code_bias: code_bias(num_bits),
            rho_sq: 0.0,
            unorm_sq: 0.0,
            ip: 0.0,
            ip_cent: 0.0,
        }
    }

    /// Adds dimension `dim`: its rotated residual and the sign bit and ex code
    /// it was quantized to.
    pub(crate) fn observe(&mut self, dim: usize, residual: f32, sign: u8, ex_code: u8) {
        let code = ((sign as u32) << self.ex_bits) + ex_code as u32;
        let centered = code as f32 - self.code_bias;
        self.rho_sq += residual * residual;
        self.unorm_sq += centered * centered;
        self.ip += residual * centered;
        self.ip_cent += residual * self.centroid[dim];
    }

    pub(crate) fn finish(self) -> SymFactors {
        let rho = self.rho_sq.sqrt();
        let unorm = self.unorm_sq.sqrt();
        let ip_cent = self.ip_cent + 0.5 * self.centroid_norm_sq;
        // A zero residual has no direction; `sym_dist` never reads its code.
        if rho == 0.0 {
            return SymFactors {
                rho,
                gamma: 1.0,
                unorm: 1.0,
                ip_cent,
            };
        }
        SymFactors {
            rho,
            gamma: self.ip / (rho * unorm),
            unorm,
            ip_cent,
        }
    }
}

/// One row as [`sym_dist`] reads it: sign words, blocked ex codes, factors.
#[derive(Debug, Clone, Copy)]
pub struct SymCode<'a> {
    pub signs: &'a [u8],
    pub ex: &'a [u8],
    pub factors: SymFactors,
}

fn split_sign_bit(signs: &[u8], dim: usize) -> u32 {
    let offset = (dim / 64) * 8;
    let word = u64::from_le_bytes(
        signs[offset..offset + 8]
            .try_into()
            .expect("8-byte sign word"),
    );
    ((word >> (63 - (dim % 64))) & 1) as u32
}

fn split_total_code(signs: &[u8], ex: &[u8], dim: usize, num_bits: u8) -> u32 {
    let ex_bits = num_bits - 1;
    let ex_code = if ex_bits == 0 {
        0
    } else {
        crate::vector::bq::ex_dot::blocked_ex_code_value(ex, dim, ex_bits) as u32
    };
    ex_code + (split_sign_bit(signs, dim) << ex_bits)
}

fn code_bias(num_bits: u8) -> f32 {
    ((1u32 << num_bits) - 1) as f32 / 2.0
}

type CenteredIpFn = fn(&[u8], &[u8], &[u8], &[u8], usize, u8) -> f32;

static CENTERED_IP: LazyLock<CenteredIpFn> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return x86::centered_code_ip;
        }
    }
    centered_code_ip_scalar
});

/// `Σ (ua - cb)(ub - cb)` from `Σ ua ub` and the code sums. `i64` keeps
/// 9-bit codes of high-dimensional vectors from overflowing.
fn finish_centered(ip_uu: i64, sum_a: i64, sum_b: i64, dim: usize, cb: f32) -> f32 {
    let center = cb as f64;
    ((ip_uu as f64) - (center * (sum_a + sum_b) as f64) + (dim as f64 * center * center)) as f32
}

fn centered_code_ip_scalar(
    signs_a: &[u8],
    ex_a: &[u8],
    signs_b: &[u8],
    ex_b: &[u8],
    dim: usize,
    num_bits: u8,
) -> f32 {
    if num_bits == 5 && dim.is_multiple_of(64) {
        return centered_code_ip_scalar_4ex(signs_a, ex_a, signs_b, ex_b, dim);
    }
    let mut ip_uu = 0i64;
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    for i in 0..dim {
        let ua = split_total_code(signs_a, ex_a, i, num_bits) as i64;
        let ub = split_total_code(signs_b, ex_b, i, num_bits) as i64;
        ip_uu += ua * ub;
        sum_a += ua;
        sum_b += ub;
    }
    finish_centered(ip_uu, sum_a, sum_b, dim, code_bias(num_bits))
}

fn centered_code_ip_scalar_4ex(
    signs_a: &[u8],
    ex_a: &[u8],
    signs_b: &[u8],
    ex_b: &[u8],
    dim: usize,
) -> f32 {
    let mut ip_uu = 0i64;
    let mut sum_a = 0i64;
    let mut sum_b = 0i64;
    for w in 0..dim / 64 {
        let word = |signs: &[u8]| {
            u64::from_le_bytes(
                signs[w * 8..w * 8 + 8]
                    .try_into()
                    .expect("8-byte sign word"),
            )
        };
        let (sa, sb) = (word(signs_a), word(signs_b));
        let pa = &ex_a[w * 32..w * 32 + 32];
        let pb = &ex_b[w * 32..w * 32 + 32];
        for g in 0..4 {
            let ga = &pa[g * 8..g * 8 + 8];
            let gb = &pb[g * 8..g * 8 + 8];
            let shift0 = 63 - (g * 16);
            for i in 0..8 {
                let s0 = ((sa >> (shift0 - i)) & 1) as u32;
                let t0 = ((sb >> (shift0 - i)) & 1) as u32;
                let s1 = ((sa >> (shift0 - 8 - i)) & 1) as u32;
                let t1 = ((sb >> (shift0 - 8 - i)) & 1) as u32;
                let ua0 = (ga[i] as u32 & 0x0f) + (s0 << 4);
                let ub0 = (gb[i] as u32 & 0x0f) + (t0 << 4);
                let ua1 = (ga[i] as u32 >> 4) + (s1 << 4);
                let ub1 = (gb[i] as u32 >> 4) + (t1 << 4);
                ip_uu += (ua0 * ub0 + ua1 * ub1) as i64;
                sum_a += (ua0 + ua1) as i64;
                sum_b += (ub0 + ub1) as i64;
            }
        }
    }
    finish_centered(ip_uu, sum_a, sum_b, dim, 15.5)
}

/// `⟨ũa, ũb⟩` of two codes centered at `((1 << num_bits) - 1) / 2`.
fn centered_code_ip(a: &SymCode, b: &SymCode, dim: usize, num_bits: u8) -> f32 {
    debug_assert_eq!(a.signs.len(), sign_word_bytes(dim));
    debug_assert_eq!(b.signs.len(), sign_word_bytes(dim));
    debug_assert!(
        num_bits == 1
            || a.ex.len() == crate::vector::bq::ex_dot::blocked_ex_code_bytes(dim, num_bits - 1)
    );
    CENTERED_IP(a.signs, a.ex, b.signs, b.ex, dim, num_bits)
}

/// Lib `sym_dist_split`: the L2 or Dot distance between two rows of the same
/// partition, estimated from their codes alone. Any metric other than Dot is
/// scored as L2, the metric a normalized Cosine index stores.
pub fn sym_dist(
    a: &SymCode,
    b: &SymCode,
    dim: usize,
    num_bits: u8,
    distance_type: DistanceType,
) -> f32 {
    let (fa, fb) = (&a.factors, &b.factors);
    if fa.rho == 0.0 && fb.rho == 0.0 {
        return match distance_type {
            DistanceType::Dot => 1.0 - (fa.ip_cent + fb.ip_cent),
            _ => 0.0,
        };
    }
    // Keep Lib's operation order: the L2 terms cancel for close rows, so a
    // reordering shows up in the result.
    let ip_bar = centered_code_ip(a, b, dim, num_bits) / (fa.unorm * fb.unorm);
    let hat = ip_bar / (fa.gamma * fb.gamma);
    match distance_type {
        DistanceType::Dot => 1.0 - (fa.ip_cent + fb.ip_cent + fa.rho * fb.rho * hat),
        _ => fa.rho * fa.rho + fb.rho * fb.rho - 2.0 * fa.rho * fb.rho * hat,
    }
}

/// Lib `SplitSingleQuery::kNumBits`. Query is 4-bit; data stays 1-bit.
pub const QUERY_WARMUP_BITS: u8 = 4;
const _: () = assert!(QUERY_WARMUP_BITS == 4);

/// `dim` rounded up to a 64-dim word, matching Lib `padded_dim`.
pub fn padded_code_dim(dim: usize) -> usize {
    dim.div_ceil(64) * 64
}

/// `u64` words in a transposed 4-bit query: one word per 64 dims per bitplane.
fn query_plane_words(dim: usize) -> usize {
    (padded_code_dim(dim) / 64) * QUERY_WARMUP_BITS as usize
}

/// 4-bit reconstruction codes plus `delta` / `vl` from Lib `quantize_scalar`.
#[derive(Debug, Clone)]
pub struct ScalarCode {
    pub codes: Vec<u8>,
    pub delta: f32,
    pub vl: f32,
    /// L2 norm of the reconstruction error `delta * code + vl - query`.
    pub error_norm: f32,
}

/// Query state for Residual `distance(id)`: BE planes + Lib `k1xsumq`.
#[derive(Debug, Clone)]
pub struct ResidualWarmupQuery {
    /// Query 4-bit planes. dim<512 is Lib block-major; dim>=512 is 512-plane-major.
    pub planes: Vec<u64>,
    pub delta: f32,
    pub vl: f32,
    pub k1xsumq: f32,
    /// Upper bound on `|warmup IP - exact 1-bit IP|` for any binary code.
    pub ip_error_bound: f32,
}

const QUERY_EX_BITS: u8 = QUERY_WARMUP_BITS - 1;
const T_CONST_SAMPLES: usize = 100;
const T_CONST_SEED: u64 = 42;
const TIGHT_START_3EX: f64 = 0.52;

/// `get_const_scaling_factors(padded_dim, 3)` used by `faster_config(dim, 4)`.
///
/// Sampling this is expensive, so callers resolve it once per index rather
/// than once per query.
pub fn const_scaling_factor_3ex(padded_dim: usize) -> f64 {
    if padded_dim == 0 {
        return 0.0;
    }
    let normal = Normal::new(0.0, 1.0).expect("unit gaussian");
    let mut rng = StdRng::seed_from_u64(T_CONST_SEED);
    let mut sum = 0.0;
    let mut row = vec![0.0f64; padded_dim];
    for _ in 0..T_CONST_SAMPLES {
        let mut norm_sq = 0.0;
        for value in row.iter_mut() {
            *value = normal.sample(&mut rng);
            norm_sq += *value * *value;
        }
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            continue;
        }
        for value in row.iter_mut() {
            *value = value.abs() / norm;
        }
        sum += best_rescale_factor_3ex(&row);
    }
    sum / T_CONST_SAMPLES as f64
}

fn quantized_level_at_scale(magnitude: f64, t: f64, max_code: i32) -> i32 {
    if magnitude == 0.0 {
        return 0;
    }
    let mut code = (t * magnitude).min(max_code as f64) as i32;
    if code < max_code && (code as f64 + 1.0) / magnitude <= t {
        code += 1;
    } else if code > 0 && code as f64 / magnitude > t {
        code -= 1;
    }
    code
}

fn best_rescale_factor_3ex(o_abs: &[f64]) -> f64 {
    let dim = o_abs.len();
    if dim == 0 {
        return 0.0;
    }
    let max_o = o_abs.iter().copied().fold(0.0, f64::max);
    if max_o == 0.0 {
        return 0.0;
    }
    let max_code = ((1u32 << QUERY_EX_BITS) - 1) as i32;
    let t_end = (max_code as f64 + 10.0) / max_o;
    let t_start = t_end * TIGHT_START_3EX;
    let mut next_t: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let mut cur_o_bar = vec![0i32; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0;
    let enqueue = |next_t: &mut BinaryHeap<Reverse<(u64, usize)>>, i: usize, code: i32| {
        let magnitude = o_abs[i];
        if magnitude > 0.0 && code < max_code {
            let next = (code as f64 + 1.0) / magnitude;
            if next < t_end {
                next_t.push(Reverse((next.to_bits(), i)));
            }
        }
    };
    for i in 0..dim {
        let cur = quantized_level_at_scale(o_abs[i], t_start, max_code);
        cur_o_bar[i] = cur;
        sqr_denominator += (cur * cur + cur) as f64;
        numerator += (cur as f64 + 0.5) * o_abs[i];
        enqueue(&mut next_t, i, cur);
    }
    let mut max_ip = numerator / sqr_denominator.sqrt();
    let mut best_t = t_start;
    while let Some(Reverse((t_bits, _))) = next_t.peek().copied() {
        let cur_t = f64::from_bits(t_bits);
        while let Some(Reverse((peek_bits, i))) = next_t.peek().copied() {
            if peek_bits != t_bits {
                break;
            }
            next_t.pop();
            cur_o_bar[i] += 1;
            sqr_denominator += 2.0 * cur_o_bar[i] as f64;
            numerator += o_abs[i];
            enqueue(&mut next_t, i, cur_o_bar[i]);
        }
        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = cur_t;
        }
    }
    best_t
}

/// Lib `quantize_scalar` with `RECONSTRUCTION` and a precomputed `t_const`.
fn quantize_scalar_reconstruct(query: &[f32], t_const: f64) -> ScalarCode {
    let dim = padded_code_dim(query.len());
    let mut residual = vec![0.0f32; dim];
    residual[..query.len()].copy_from_slice(query);
    let norm = residual
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm == 0.0 {
        return ScalarCode {
            codes: vec![0; dim],
            delta: 0.0,
            vl: 0.0,
            error_norm: 0.0,
        };
    }
    let max_code = (1u32 << QUERY_EX_BITS) - 1;
    let mut codes = vec![0u8; dim];
    for i in 0..dim {
        let abs_n = residual[i].abs() / norm;
        let mut code = ((t_const * abs_n as f64) + 1e-5) as i32;
        if code < 0 {
            code = 0;
        }
        if code as u32 > max_code {
            code = max_code as i32;
        }
        if residual[i] > 0.0 {
            code += 1 << QUERY_EX_BITS;
        } else {
            // Lib `ex_bits_code`: r<=0 → `(~tmp) & ((1<<ex_bits)-1)`.
            code = (!code) & (max_code as i32);
        }
        codes[i] = code as u8;
    }
    let cb = -((1u32 << QUERY_EX_BITS) as f32 - 0.5);
    let mut dot = 0.0f32;
    let mut quan_sq = 0.0f32;
    for i in 0..dim {
        let u_cb = codes[i] as f32 + cb;
        dot += residual[i] * u_cb;
        quan_sq += u_cb * u_cb;
    }
    let norm_quan = quan_sq.sqrt();
    let cos = dot / (norm * norm_quan);
    let delta = norm / norm_quan * cos;
    ScalarCode {
        codes,
        delta,
        vl: delta * cb,
        // `delta` projects the query onto the code direction, leaving the
        // orthogonal part as the error.
        error_norm: norm * (1.0 - cos * cos).max(0.0).sqrt(),
    }
}

/// 64-dim words in one Lib `new_transpose_bin_512` superblock.
const WARMUP_SUPER_CHUNKS: usize = 8;

/// Lib `new_transpose_bin_512`: dim0 at bit 63; each 512-dim superblock is
/// plane-major (`tq[j * nchunk + k]`).
fn transpose_query_bin_be(codes: &[u8], out: &mut [u64]) {
    debug_assert_eq!(out.len(), query_plane_words(codes.len()));
    out.fill(0);
    let bits = QUERY_WARMUP_BITS as usize;
    let padded = padded_code_dim(codes.len());
    let mut dim = 0;
    let mut out_off = 0;
    while dim < padded {
        let nchunk = ((padded - dim) / 64).min(WARMUP_SUPER_CHUNKS);
        for k in 0..nchunk {
            let start = dim + k * 64;
            for offset in 0..64 {
                let idx = start + offset;
                if idx >= codes.len() {
                    break;
                }
                let code = codes[idx];
                for j in 0..bits {
                    if (code >> j) & 1 == 1 {
                        out[out_off + j * nchunk + k] |= 1u64 << (63 - offset);
                    }
                }
            }
        }
        out_off += nchunk * bits;
        dim += nchunk * 64;
    }
}

/// Lib scalar `warmup_ip_x0_q` layout: `[block0 planes 0..3, block1 planes 0..3, ...]`.
fn transpose_query_bin_block_major(codes: &[u8], out: &mut [u64]) {
    debug_assert_eq!(out.len(), query_plane_words(codes.len()));
    out.fill(0);
    let bits = QUERY_WARMUP_BITS as usize;
    let padded = padded_code_dim(codes.len());
    let nblk = padded / 64;
    for k in 0..nblk {
        let start = k * 64;
        for offset in 0..64 {
            let idx = start + offset;
            if idx >= codes.len() {
                break;
            }
            let code = codes[idx];
            for j in 0..bits {
                if (code >> j) & 1 == 1 {
                    out[k * bits + j] |= 1u64 << (63 - offset);
                }
            }
        }
    }
}

fn transpose_query_for_warmup(codes: &[u8], out: &mut [u64]) {
    if padded_code_dim(codes.len()) >= 512 {
        transpose_query_bin_be(codes, out);
    } else {
        transpose_query_bin_block_major(codes, out);
    }
}

#[inline(always)]
fn warmup_ip_accumulate_block_major(data_be: &[u8], query: &[u64]) -> (u64, u64) {
    let num_blk = data_be.len() / 8;
    debug_assert_eq!(query.len(), num_blk * QUERY_WARMUP_BITS as usize);
    let data = data_be.as_ptr() as *const u64;
    let query = query.as_ptr();
    let mut ip = 0u64;
    let mut ppc = 0u64;
    // SAFETY: `data_be` is `num_blk` little-endian u64 words; query is block-major.
    unsafe {
        for k in 0..num_blk {
            let x = data.add(k).read_unaligned();
            ppc += x.count_ones() as u64;
            let q = query.add(k * QUERY_WARMUP_BITS as usize);
            ip += (x & *q).count_ones() as u64;
            ip += ((x & *q.add(1)).count_ones() as u64) << 1;
            ip += ((x & *q.add(2)).count_ones() as u64) << 2;
            ip += ((x & *q.add(3)).count_ones() as u64) << 3;
        }
    }
    (ip, ppc)
}

#[inline(always)]
fn warmup_ip_accumulate_scalar(data_be: &[u8], query_planes: &[u64]) -> (u64, u64) {
    let num_blk = data_be.len() / 8;
    debug_assert_eq!(query_planes.len(), num_blk * QUERY_WARMUP_BITS as usize);
    let data = data_be.as_ptr() as *const u64;
    let query = query_planes.as_ptr();
    let mut ip = 0u64;
    let mut ppc = 0u64;
    let mut blk = 0;
    let mut q_off = 0;
    while blk < num_blk {
        let nchunk = (num_blk - blk).min(WARMUP_SUPER_CHUNKS);
        // SAFETY: `data_be` is `num_blk` little-endian u64 words; query is
        // plane-major inside each 512-dim superblock (same as Lib).
        unsafe {
            for k in 0..nchunk {
                let x = data.add(blk + k).read_unaligned();
                ppc += x.count_ones() as u64;
                ip += (x & *query.add(q_off + k)).count_ones() as u64;
                ip += ((x & *query.add(q_off + nchunk + k)).count_ones() as u64) << 1;
                ip += ((x & *query.add(q_off + 2 * nchunk + k)).count_ones() as u64) << 2;
                ip += ((x & *query.add(q_off + 3 * nchunk + k)).count_ones() as u64) << 3;
            }
        }
        blk += nchunk;
        q_off += nchunk * QUERY_WARMUP_BITS as usize;
    }
    (ip, ppc)
}

/// Lib scalar `warmup_ip_x0_q`.
#[inline(always)]
fn warmup_ip_x0_q_scalar(data_be: &[u8], query_planes: &[u64], delta: f32, vl: f32) -> f32 {
    debug_assert!(data_be.len().is_multiple_of(8));
    debug_assert_eq!(
        query_planes.len(),
        (data_be.len() / 8) * QUERY_WARMUP_BITS as usize
    );
    let (ip, ppc) = if data_be.len() >= WARMUP_SUPER_CHUNKS * 8 {
        warmup_ip_accumulate_scalar(data_be, query_planes)
    } else {
        warmup_ip_accumulate_block_major(data_be, query_planes)
    };
    delta * ip as f32 + vl * ppc as f32
}

/// Lib `warmup_ip_x0_q_512` when dim>=512 and AVX2 exist; else Lib scalar POPCNT.
#[inline(always)]
pub fn warmup_ip_x0_q(data_be: &[u8], query_planes: &[u64], delta: f32, vl: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 is a 512-dim kernel. Below that, hardware POPCNT on block-major
        // words beats AVX2 popcount variants: Lib's maskload remainder loses at
        // dim=128, and a per-block broadcast/AND/`popcount_avx2`/sllv kernel
        // loses ~25-50% across dim=64..384 (benches/sym_dist.rs warmup_scan).
        if data_be.len() >= WARMUP_SUPER_CHUNKS * 8 && std::is_x86_feature_detected!("avx2") {
            return unsafe { warmup_avx2::warmup_ip_x0_q_512(data_be, query_planes, delta, vl) };
        }
    }
    warmup_ip_x0_q_scalar(data_be, query_planes, delta, vl)
}

#[cfg(target_arch = "x86_64")]
mod warmup_avx2 {
    use super::QUERY_WARMUP_BITS;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn popcount_avx2(v: __m256i) -> __m256i {
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        _mm256_sad_epu8(
            _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo),
                _mm256_shuffle_epi8(lookup, hi),
            ),
            _mm256_setzero_si256(),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mm256_reduce_add_epi64(v: __m256i) -> u64 {
        let low = _mm256_castsi256_si128(v);
        let high = _mm256_extracti128_si256::<1>(v);
        let sum = _mm_add_epi64(low, high);
        _mm_extract_epi64::<0>(sum) as u64 + _mm_extract_epi64::<1>(sum) as u64
    }

    /// Lib `warmup_ip_x0_q_512_avx2`. Caller must have checked AVX2 and dim>=512.
    #[target_feature(enable = "avx2")]
    pub unsafe fn warmup_ip_x0_q_512(data_be: &[u8], query: &[u64], delta: f32, vl: f32) -> f32 {
        debug_assert!(data_be.len().is_multiple_of(8));
        let num_blk = data_be.len() / 8;
        debug_assert_eq!(query.len(), num_blk * QUERY_WARMUP_BITS as usize);
        let padded_dim = num_blk * 64;
        let mut data_ptr = data_be.as_ptr() as *const u64;
        let mut query_ptr = query.as_ptr();

        let mut acc_ip = _mm256_setzero_si256();
        let mut acc_ppc = _mm256_setzero_si256();
        let mut acc_bits = [_mm256_setzero_si256(); 4];

        let dim_end_512 = (padded_dim / 512) * 512;
        let mut dim = 0;
        while dim < dim_end_512 {
            let data_lo = _mm256_loadu_si256(data_ptr as *const __m256i);
            let data_hi = _mm256_loadu_si256(data_ptr.add(4) as *const __m256i);
            data_ptr = data_ptr.add(8);
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_lo));
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_hi));
            for acc in acc_bits.iter_mut() {
                let q_lo = _mm256_loadu_si256(query_ptr as *const __m256i);
                let q_hi = _mm256_loadu_si256(query_ptr.add(4) as *const __m256i);
                query_ptr = query_ptr.add(8);
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_lo, q_lo)));
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_hi, q_hi)));
            }
            dim += 512;
        }

        let remaining = padded_dim - dim;
        if remaining > 0 {
            let num_chunks_64 = remaining / 64;
            let num_chunks_32 = remaining / 32;
            let chunks_lo = num_chunks_32.min(8) as i32;
            let chunks_hi = num_chunks_32.saturating_sub(8) as i32;
            let sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            let mask_lo = _mm256_cmpgt_epi32(_mm256_set1_epi32(chunks_lo), sequence);
            let mask_hi = _mm256_cmpgt_epi32(_mm256_set1_epi32(chunks_hi), sequence);
            // SAFETY: `add(4)` is only formed when ≥5 leftover u64 words so the
            // high 256-bit pointer stays inside the allocation. Lib always
            // computes `data+4`; that is UB in Rust when leftover is 1..=3.
            let data_lo = _mm256_maskload_epi32(data_ptr as *const i32, mask_lo);
            let data_hi = if num_chunks_64 > 4 {
                _mm256_maskload_epi32(data_ptr.add(4) as *const i32, mask_hi)
            } else {
                _mm256_setzero_si256()
            };
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_lo));
            acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_hi));
            for acc in acc_bits.iter_mut() {
                let q_lo = _mm256_maskload_epi32(query_ptr as *const i32, mask_lo);
                let q_hi = if num_chunks_64 > 4 {
                    _mm256_maskload_epi32(query_ptr.add(4) as *const i32, mask_hi)
                } else {
                    _mm256_setzero_si256()
                };
                query_ptr = query_ptr.add(num_chunks_64);
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_lo, q_lo)));
                *acc = _mm256_add_epi64(*acc, popcount_avx2(_mm256_and_si256(data_hi, q_hi)));
            }
        }

        for (j, acc) in acc_bits.iter().enumerate() {
            let shift = _mm_cvtsi32_si128(j as i32);
            acc_ip = _mm256_add_epi64(acc_ip, _mm256_sll_epi64(*acc, shift));
        }
        let ip = mm256_reduce_add_epi64(acc_ip);
        let ppc = mm256_reduce_add_epi64(acc_ppc);
        delta * ip as f32 + vl * ppc as f32
    }
}

/// Quantize + transpose a rotated residual query for Residual HNSW search.
/// `t_const` comes from [`const_scaling_factor_3ex`] for the padded dimension.
pub fn prepare_residual_warmup_query(rotated: &[f32], t_const: f64) -> ResidualWarmupQuery {
    let quantized = quantize_scalar_reconstruct(rotated, t_const);
    let mut planes = vec![0u64; query_plane_words(rotated.len())];
    transpose_query_for_warmup(&quantized.codes, &mut planes);
    let dim = rotated.len();
    let sum_q = rotated.iter().copied().sum::<f32>();
    let code_sum = quantized.codes[..dim]
        .iter()
        .map(|&code| code as u32)
        .sum::<u32>();
    let sum_error = quantized.delta * code_sum as f32 + quantized.vl * dim as f32 - sum_q;
    ResidualWarmupQuery {
        planes,
        delta: quantized.delta,
        vl: quantized.vl,
        k1xsumq: sum_q * -0.5,
        // A binary code `b` sums the reconstruction error `err` over its set
        // bits, `(sum(err) + <2b - 1, err>) / 2`, and padded bits are never
        // set; Cauchy-Schwarz bounds `|<2b - 1, err>|` by `sqrt(dim) |err|`.
        ip_error_bound: 0.5 * (sum_error.abs() + (dim as f32).sqrt() * quantized.error_norm),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    const fn sign_expand_u8(bits: u8) -> u64 {
        let mut word = 0u64;
        let mut i = 0u32;
        while i < 8 {
            if (bits >> (7 - i)) & 1 == 1 {
                word |= 16u64 << (8 * i);
            }
            i += 1;
        }
        word
    }

    const fn make_sign_lut() -> [u64; 256] {
        let mut lut = [0u64; 256];
        let mut i = 0;
        while i < 256 {
            lut[i] = sign_expand_u8(i as u8);
            i += 1;
        }
        lut
    }

    const SIGN_EXPAND: [u64; 256] = make_sign_lut();

    const fn bit_expand_le_u8(bits: u8) -> u64 {
        let mut word = 0u64;
        let mut i = 0u32;
        while i < 8 {
            if (bits >> i) & 1 == 1 {
                word |= 1u64 << (8 * i);
            }
            i += 1;
        }
        word
    }

    const fn make_bit_expand_lut() -> [u64; 256] {
        let mut lut = [0u64; 256];
        let mut i = 0;
        while i < 256 {
            lut[i] = bit_expand_le_u8(i as u8);
            i += 1;
        }
        lut
    }

    /// Lib `kBitExpandLe`: byte `i` of the word is bit `i` of the input byte.
    const BIT_EXPAND_LE: [u64; 256] = make_bit_expand_lut();

    #[inline(always)]
    unsafe fn word_at(bin: *const u8, w: usize) -> u64 {
        // SAFETY: caller sized `bin` to at least `(w + 1) * 8` bytes.
        (bin as *const u64).add(w).read_unaligned()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_4bit_xmm(raw8: __m128i) -> __m128i {
        let lo_mask = _mm_set1_epi8(0x0f);
        let lo = _mm_and_si128(raw8, lo_mask);
        let hi = _mm_and_si128(
            _mm_srli_epi16(_mm_and_si128(raw8, _mm_set1_epi8(0xf0u8 as i8)), 4),
            lo_mask,
        );
        _mm_unpacklo_epi64(lo, hi)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signs_to_bytes(sign_bits: u32) -> __m128i {
        _mm_set_epi64x(
            SIGN_EXPAND[(sign_bits & 255) as usize] as i64,
            SIGN_EXPAND[(sign_bits >> 8) as usize] as i64,
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_u16(packed: *const u8, sign_bits: u32) -> __m128i {
        let raw = _mm_loadl_epi64(packed as *const __m128i);
        _mm_add_epi8(unpack_4bit_xmm(raw), signs_to_bytes(sign_bits))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn acc_16(
        ua: __m128i,
        ub: __m128i,
        ones: __m128i,
        zero: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        *ip = _mm_add_epi32(*ip, _mm_madd_epi16(_mm_maddubs_epi16(ua, ub), ones));
        *sum_a = _mm_add_epi64(*sum_a, _mm_sad_epu8(ua, zero));
        *sum_b = _mm_add_epi64(*sum_b, _mm_sad_epu8(ub, zero));
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_epi32_128(v: __m128i) -> i32 {
        let y = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4e));
        let z = _mm_add_epi32(y, _mm_shuffle_epi32(y, 0xb1));
        _mm_cvtsi128_si32(z)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_sad_128(v: __m128i) -> i32 {
        _mm_cvtsi128_si32(v) + _mm_extract_epi32(v, 2)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn acc_word_4bit(
        sa: u64,
        sb: u64,
        pa: *const u8,
        pb: *const u8,
        ones: __m128i,
        zero: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        acc_16(
            decode_u16(pa, (sa >> 48) as u32),
            decode_u16(pb, (sb >> 48) as u32),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_u16(pa.add(8), ((sa >> 32) as u32) & 0xffff),
            decode_u16(pb.add(8), ((sb >> 32) as u32) & 0xffff),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_u16(pa.add(16), ((sa >> 16) as u32) & 0xffff),
            decode_u16(pb.add(16), ((sb >> 16) as u32) & 0xffff),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_u16(pa.add(24), sa as u32 & 0xffff),
            decode_u16(pb.add(24), sb as u32 & 0xffff),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_2bit(field: __m128i, sign_bits: u32) -> __m128i {
        _mm_add_epi8(field, _mm_srli_epi16(signs_to_bytes(sign_bits), 2))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn acc_word_2bit(
        sa: u64,
        sb: u64,
        pa: *const u8,
        pb: *const u8,
        ones: __m128i,
        zero: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        // SAFETY: each 64-dim extra block is 16 bytes for 2-bit codes.
        let raw_a = _mm_loadu_si128(pa as *const __m128i);
        let raw_b = _mm_loadu_si128(pb as *const __m128i);
        let m03 = _mm_set1_epi8(3);
        let m0c = _mm_set1_epi8(0x0c);
        let m30 = _mm_set1_epi8(0x30);
        let mc0 = _mm_set1_epi8(0xc0u8 as i8);
        acc_16(
            decode_2bit(_mm_and_si128(raw_a, m03), (sa >> 48) as u32),
            decode_2bit(_mm_and_si128(raw_b, m03), (sb >> 48) as u32),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_a, m0c), 2),
                ((sa >> 32) as u32) & 0xffff,
            ),
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_b, m0c), 2),
                ((sb >> 32) as u32) & 0xffff,
            ),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_a, m30), 4),
                ((sa >> 16) as u32) & 0xffff,
            ),
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_b, m30), 4),
                ((sb >> 16) as u32) & 0xffff,
            ),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
        acc_16(
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_a, mc0), 6),
                sa as u32 & 0xffff,
            ),
            decode_2bit(
                _mm_srli_epi16(_mm_and_si128(raw_b, mc0), 6),
                sb as u32 & 0xffff,
            ),
            ones,
            zero,
            ip,
            sum_a,
            sum_b,
        );
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_2bit(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
    ) -> f32 {
        let ones = _mm_set1_epi16(1);
        let zero = _mm_setzero_si128();
        let mut ip = zero;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
            acc_word_2bit(
                word_at(bin_a.as_ptr(), w),
                word_at(bin_b.as_ptr(), w),
                ex_a.as_ptr().add(w * 16),
                ex_b.as_ptr().add(w * 16),
                ones,
                zero,
                &mut ip,
                &mut sum_a,
                &mut sum_b,
            );
        }
        super::finish_centered(
            hsum_epi32_128(ip) as i64,
            hsum_sad_128(sum_a) as i64,
            hsum_sad_128(sum_b) as i64,
            dim,
            3.5,
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_4bit(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
    ) -> f32 {
        let ones = _mm_set1_epi16(1);
        let zero = _mm_setzero_si128();
        let mut ip = zero;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
            acc_word_4bit(
                word_at(bin_a.as_ptr(), w),
                word_at(bin_b.as_ptr(), w),
                ex_a.as_ptr().add(w * 32),
                ex_b.as_ptr().add(w * 32),
                ones,
                zero,
                &mut ip,
                &mut sum_a,
                &mut sum_b,
            );
        }
        super::finish_centered(
            hsum_epi32_128(ip) as i64,
            hsum_sad_128(sum_a) as i64,
            hsum_sad_128(sum_b) as i64,
            dim,
            15.5,
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn acc_u8_products(a: __m128i, b: __m128i, u16_mask: __m128i, ip_xy: &mut __m128i) {
        let p0 = _mm_mullo_epi16(_mm_cvtepu8_epi16(a), _mm_cvtepu8_epi16(b));
        *ip_xy = _mm_add_epi32(
            *ip_xy,
            _mm_add_epi32(_mm_and_si128(p0, u16_mask), _mm_srli_epi32(p0, 16)),
        );
        let p1 = _mm_mullo_epi16(
            _mm_cvtepu8_epi16(_mm_srli_si128(a, 8)),
            _mm_cvtepu8_epi16(_mm_srli_si128(b, 8)),
        );
        *ip_xy = _mm_add_epi32(
            *ip_xy,
            _mm_add_epi32(_mm_and_si128(p1, u16_mask), _mm_srli_epi32(p1, 16)),
        );
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn acc_16_u8(
        xa: *const u8,
        xb: *const u8,
        sa: u32,
        sb: u32,
        zero: __m128i,
        sign_on: __m128i,
        u16_mask: __m128i,
        ip_xy: &mut __m128i,
        sum_x: &mut __m128i,
        sum_y: &mut __m128i,
        sum_sa_xb: &mut __m128i,
        sum_sb_xa: &mut __m128i,
    ) {
        // SAFETY: each 8-bit extra run is 16 bytes.
        let a = _mm_loadu_si128(xa as *const __m128i);
        let b = _mm_loadu_si128(xb as *const __m128i);
        let ma = _mm_cmpeq_epi8(signs_to_bytes(sa), sign_on);
        let mb = _mm_cmpeq_epi8(signs_to_bytes(sb), sign_on);
        acc_u8_products(a, b, u16_mask, ip_xy);
        *sum_x = _mm_add_epi64(*sum_x, _mm_sad_epu8(a, zero));
        *sum_y = _mm_add_epi64(*sum_y, _mm_sad_epu8(b, zero));
        *sum_sa_xb = _mm_add_epi64(*sum_sa_xb, _mm_sad_epu8(_mm_and_si128(b, ma), zero));
        *sum_sb_xa = _mm_add_epi64(*sum_sb_xa, _mm_sad_epu8(_mm_and_si128(a, mb), zero));
    }

    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_8bit(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
    ) -> f32 {
        let zero = _mm_setzero_si128();
        let sign_on = _mm_set1_epi8(16);
        let u16_mask = _mm_set1_epi32(0x0000_ffff);
        let mut ip_xy = zero;
        let mut sum_x = zero;
        let mut sum_y = zero;
        let mut sum_sa_xb = zero;
        let mut sum_sb_xa = zero;
        let mut n_sa = 0i32;
        let mut n_sb = 0i32;
        let mut n_and = 0i32;
        // Buffers are zero-padded to whole 64-dim blocks, so iterate by block
        // count; the zero tail contributes nothing to the accumulators.
        let words = bin_a.len() / 8;
        for w in 0..words {
            let sa = word_at(bin_a.as_ptr(), w);
            let sb = word_at(bin_b.as_ptr(), w);
            let pa = ex_a.as_ptr().add(w * 64);
            let pb = ex_b.as_ptr().add(w * 64);
            let s0 = (sa >> 48) as u32;
            let t0 = (sb >> 48) as u32;
            let s1 = ((sa >> 32) as u32) & 0xffff;
            let t1 = ((sb >> 32) as u32) & 0xffff;
            let s2 = ((sa >> 16) as u32) & 0xffff;
            let t2 = ((sb >> 16) as u32) & 0xffff;
            let s3 = sa as u32 & 0xffff;
            let t3 = sb as u32 & 0xffff;
            acc_16_u8(
                pa,
                pb,
                s0,
                t0,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            acc_16_u8(
                pa.add(16),
                pb.add(16),
                s1,
                t1,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            acc_16_u8(
                pa.add(32),
                pb.add(32),
                s2,
                t2,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            acc_16_u8(
                pa.add(48),
                pb.add(48),
                s3,
                t3,
                zero,
                sign_on,
                u16_mask,
                &mut ip_xy,
                &mut sum_x,
                &mut sum_y,
                &mut sum_sa_xb,
                &mut sum_sb_xa,
            );
            n_sa += sa.count_ones() as i32;
            n_sb += sb.count_ones() as i32;
            n_and += (sa & sb).count_ones() as i32;
        }
        let ip_uu = hsum_epi32_128(ip_xy) as i64
            + 256 * (hsum_sad_128(sum_sa_xb) + hsum_sad_128(sum_sb_xa)) as i64
            + 65536 * n_and as i64;
        let sum_a = hsum_sad_128(sum_x) as i64 + 256 * n_sa as i64;
        let sum_b = hsum_sad_128(sum_y) as i64 + 256 * n_sb as i64;
        super::finish_centered(ip_uu, sum_a, sum_b, dim, 255.5)
    }

    /// Lib `extra_1bit_16`: 16 one-bit codes from a little-endian u16 word.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn extra_1bit_16(word: u16) -> __m128i {
        _mm_set_epi64x(
            BIT_EXPAND_LE[(word >> 8) as usize] as i64,
            BIT_EXPAND_LE[(word & 255) as usize] as i64,
        )
    }

    /// Lib `top_bits_group`: byte `b` holds the top bit of dim `16*group + b`
    /// at bit position `p`. The plane packs dim `i` at byte `i % 8`, bit
    /// `i / 8` of the little-endian u64.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn top_bits_group(top: u64, p: u32, group: u32) -> __m128i {
        let high_shift = p as i32 - 1 - 2 * group as i32;
        let low_shift = p as i32 - 2 * group as i32;
        let high = if high_shift >= 0 {
            top << high_shift
        } else {
            top >> (-high_shift) as u32
        };
        let low = if low_shift >= 0 {
            top << low_shift
        } else {
            top >> (-low_shift) as u32
        };
        _mm_and_si128(
            _mm_set_epi64x(high as i64, low as i64),
            _mm_set1_epi8((1u8 << p) as i8),
        )
    }

    /// Lib `sign_contrib_bytes`: each set sign bit contributes `1 << ex_bits`.
    /// Sign bytes start at 16 (`1 << 4`) and shift to the target magnitude;
    /// `ex_bits <= 7` keeps every value inside its byte lane.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sign_contrib_bytes(sign_bits: u32, ex_bits: u8) -> __m128i {
        let s16 = signs_to_bytes(sign_bits);
        if ex_bits <= 4 {
            _mm_srl_epi16(s16, _mm_cvtsi32_si128(4 - ex_bits as i32))
        } else {
            _mm_sll_epi16(s16, _mm_cvtsi32_si128(ex_bits as i32 - 4))
        }
    }

    /// Lib `unpack_2bit_fields`: byte `b` holds dims `b`, `b+16`, `b+32`,
    /// `b+48` at bit pairs 0/2/4/6.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_2bit_fields(ex: *const u8, out: &mut [__m128i; 4]) {
        // SAFETY: `ex` points to one 64-dim block of 2-bit codes (16 bytes).
        let raw = _mm_loadu_si128(ex as *const __m128i);
        let mask = _mm_set1_epi8(3);
        out[0] = _mm_and_si128(raw, mask);
        out[1] = _mm_and_si128(_mm_srli_epi16(raw, 2), mask);
        out[2] = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
        out[3] = _mm_and_si128(_mm_srli_epi16(raw, 6), mask);
    }

    /// Lib `unpack_6bit_fields`: three 16-byte chunks hold the low 6 bits of
    /// dims 0-47; dim `48+b` gathers the top 2 bits of byte `b` per chunk.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_6bit_fields(ex: *const u8, out: &mut [__m128i; 4]) {
        // SAFETY: `ex` points to one 64-dim block of 6-bit codes (48 bytes).
        let cpt1 = _mm_loadu_si128(ex as *const __m128i);
        let cpt2 = _mm_loadu_si128(ex.add(16) as *const __m128i);
        let cpt3 = _mm_loadu_si128(ex.add(32) as *const __m128i);
        let mask6 = _mm_set1_epi8(0x3f);
        let mask2 = _mm_set1_epi8(0xc0u8 as i8);
        out[0] = _mm_and_si128(cpt1, mask6);
        out[1] = _mm_and_si128(cpt2, mask6);
        out[2] = _mm_and_si128(cpt3, mask6);
        out[3] = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4),
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2),
        );
    }

    /// Lib `unpack_ex_64`: decode one 64-dim block of blocked-layout extra
    /// codes into four 16-byte groups (`out[g]` = dims `16g..16g+15`).
    ///
    /// `ex` must point to a full 64-dim block of `8 * ex_bits` bytes. Total
    /// widths 3, 5 and 9 have dedicated kernels and never get here.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_ex_64(ex: *const u8, ex_bits: u8, out: &mut [__m128i; 4]) {
        match ex_bits {
            0 => {
                *out = [_mm_setzero_si128(); 4];
            }
            1 => {
                for (group, slot) in out.iter_mut().enumerate() {
                    // SAFETY: 8 bytes per 64-dim block; each group reads 2.
                    *slot = extra_1bit_16((ex.add(group * 2) as *const u16).read_unaligned());
                }
            }
            3 => {
                unpack_2bit_fields(ex, out);
                // SAFETY: 24-byte block; reads the top-bit plane (bytes 16..24).
                let top = (ex.add(16) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 2, group as u32));
                }
            }
            5 => {
                // SAFETY: 40-byte block; two 16-byte chunks + top plane at 32..40.
                let c0 = _mm_loadu_si128(ex as *const __m128i);
                let c1 = _mm_loadu_si128(ex.add(16) as *const __m128i);
                let mask = _mm_set1_epi8(0x0f);
                out[0] = _mm_and_si128(c0, mask);
                out[1] = _mm_and_si128(_mm_srli_epi16(c0, 4), mask);
                out[2] = _mm_and_si128(c1, mask);
                out[3] = _mm_and_si128(_mm_srli_epi16(c1, 4), mask);
                let top = (ex.add(32) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 4, group as u32));
                }
            }
            6 => unpack_6bit_fields(ex, out),
            7 => {
                unpack_6bit_fields(ex, out);
                // SAFETY: 56-byte block; reads the top-bit plane (bytes 48..56).
                let top = (ex.add(48) as *const u64).read_unaligned();
                for (group, slot) in out.iter_mut().enumerate() {
                    *slot = _mm_or_si128(*slot, top_bits_group(top, 6, group as u32));
                }
            }
            _ => unreachable!("ex_bits {ex_bits} has a dedicated kernel"),
        }
    }

    /// Lib `decode_u_64`: extra codes plus the sign-bit contribution, giving
    /// the unsigned `num_bits`-wide codes of one 64-dim block.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_u_64(signs: u64, ex: *const u8, ex_bits: u8, codes: &mut [__m128i; 4]) {
        unpack_ex_64(ex, ex_bits, codes);
        let groups = [
            (signs >> 48) as u32,
            ((signs >> 32) & 0xffff) as u32,
            ((signs >> 16) & 0xffff) as u32,
            (signs & 0xffff) as u32,
        ];
        for (code, sign_bits) in codes.iter_mut().zip(groups) {
            *code = _mm_add_epi8(*code, sign_contrib_bytes(sign_bits, ex_bits));
        }
    }

    /// Lib `acc_16_u8_full`: widening u8 products for codes above the
    /// `maddubs` signed range (total bits = 8).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn acc_16_u8_full(
        ua: __m128i,
        ub: __m128i,
        zero: __m128i,
        u16_mask: __m128i,
        ip: &mut __m128i,
        sum_a: &mut __m128i,
        sum_b: &mut __m128i,
    ) {
        acc_u8_products(ua, ub, u16_mask, ip);
        *sum_a = _mm_add_epi64(*sum_a, _mm_sad_epu8(ua, zero));
        *sum_b = _mm_add_epi64(*sum_b, _mm_sad_epu8(ub, zero));
    }

    /// Lib `centered_code_ip_fitted`: one generic AVX2 kernel for total bit
    /// widths 1-8. `maddubs` covers codes up to 127 (bits <= 7); bits = 8
    /// uses widening 16-bit products.
    #[target_feature(enable = "avx2")]
    unsafe fn centered_code_ip_fitted(
        bin_a: &[u8],
        ex_a: &[u8],
        bin_b: &[u8],
        ex_b: &[u8],
        dim: usize,
        num_bits: u8,
    ) -> f32 {
        let ex_bits = num_bits - 1;
        let stride = (64 * ex_bits as usize) / 8;
        let use_maddubs = num_bits <= 7;
        let ones = _mm_set1_epi16(1);
        let zero = _mm_setzero_si128();
        let u16_mask = _mm_set1_epi32(0x0000_ffff);
        let mut ip = zero;
        let mut sum_a = zero;
        let mut sum_b = zero;
        // Iterate padded 64-dim blocks (see centered_code_ip_8bit).
        let words = bin_a.len() / 8;
        for w in 0..words {
            let mut ua = [zero; 4];
            let mut ub = [zero; 4];
            // SAFETY: `stride` is the per-block extra-code byte count; with
            // `ex_bits == 0` the pointer is never dereferenced.
            decode_u_64(
                word_at(bin_a.as_ptr(), w),
                ex_a.as_ptr().add(w * stride),
                ex_bits,
                &mut ua,
            );
            decode_u_64(
                word_at(bin_b.as_ptr(), w),
                ex_b.as_ptr().add(w * stride),
                ex_bits,
                &mut ub,
            );
            for g in 0..4 {
                if use_maddubs {
                    acc_16(ua[g], ub[g], ones, zero, &mut ip, &mut sum_a, &mut sum_b);
                } else {
                    acc_16_u8_full(
                        ua[g], ub[g], zero, u16_mask, &mut ip, &mut sum_a, &mut sum_b,
                    );
                }
            }
        }
        super::finish_centered(
            hsum_epi32_128(ip) as i64,
            hsum_sad_128(sum_a) as i64,
            hsum_sad_128(sum_b) as i64,
            dim,
            super::code_bias(num_bits),
        )
    }

    /// AVX2 [`super::centered_code_ip`]. Both rows are zero-padded to whole
    /// 64-dim blocks, so the kernels iterate every padded block: zero codes
    /// add nothing, and `finish_centered` centers with the true `dim`.
    pub(super) fn centered_code_ip(
        signs_a: &[u8],
        ex_a: &[u8],
        signs_b: &[u8],
        ex_b: &[u8],
        dim: usize,
        num_bits: u8,
    ) -> f32 {
        // SAFETY: `CENTERED_IP` selects this function only when AVX2 is present.
        unsafe {
            match num_bits {
                3 => centered_code_ip_2bit(signs_a, ex_a, signs_b, ex_b, dim),
                5 => centered_code_ip_4bit(signs_a, ex_a, signs_b, ex_b, dim),
                9 => centered_code_ip_8bit(signs_a, ex_a, signs_b, ex_b, dim),
                _ => centered_code_ip_fitted(signs_a, ex_a, signs_b, ex_b, dim, num_bits),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use serde::Deserialize;

    use super::*;
    use crate::vector::bq::ex_dot::{
        blocked_ex_code_bytes, blocked_ex_code_value, pack_blocked_row,
    };

    /// Lib outputs: per-vector factors and pairwise `sym_dist`, with f32
    /// values as hex IEEE-754 bit patterns.
    const SYM_DIST_GOLDEN: &str = include_str!("../../../testdata/symrabitq/sym_dist.json");

    #[derive(Deserialize)]
    struct Golden {
        vectors: Vec<GoldenVector>,
        pairs: Vec<GoldenPair>,
    }

    #[derive(Deserialize)]
    struct GoldenVector {
        id: String,
        metric: String,
        num_bits: u8,
        dim: usize,
        centroid_hex: String,
        residual_hex: String,
        bin_hex: String,
        ex_hex: String,
        rho: String,
        gamma: String,
        unorm: String,
        ip_cent: String,
    }

    #[derive(Deserialize)]
    struct GoldenPair {
        a: String,
        b: String,
        sym_dist: String,
    }

    impl GoldenVector {
        fn factors(&self) -> SymFactors {
            SymFactors {
                rho: f32_from_hex(&self.rho),
                gamma: f32_from_hex(&self.gamma),
                unorm: f32_from_hex(&self.unorm),
                ip_cent: f32_from_hex(&self.ip_cent),
            }
        }

        fn distance_type(&self) -> DistanceType {
            match self.metric.as_str() {
                "l2" => DistanceType::L2,
                "dot" => DistanceType::Dot,
                other => panic!("unsupported golden metric {other}"),
            }
        }
    }

    fn golden() -> Golden {
        serde_json::from_str(SYM_DIST_GOLDEN).unwrap()
    }

    fn bytes_from_hex(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn f32_from_hex(hex: &str) -> f32 {
        f32::from_bits(u32::from_str_radix(hex, 16).unwrap())
    }

    fn f32s_from_hex(hex: &str) -> Vec<f32> {
        (0..hex.len())
            .step_by(8)
            .map(|i| f32_from_hex(&hex[i..i + 8]))
            .collect()
    }

    fn assert_close(got: f32, expected: f32, context: &str) {
        let err = (got - expected).abs() / expected.abs().max(1.0);
        assert!(err <= 1e-5, "{context}: got {got}, expected {expected}");
    }

    #[test]
    fn test_sym_factors_match_lib() {
        for vector in golden().vectors {
            let residual = f32s_from_hex(&vector.residual_hex);
            let centroid = f32s_from_hex(&vector.centroid_hex);
            let signs = bytes_from_hex(&vector.bin_hex);
            let ex = bytes_from_hex(&vector.ex_hex);
            let centroid_norm_sq = centroid.iter().map(|value| value * value).sum();
            let mut builder = SymFactorsBuilder::new(&centroid, centroid_norm_sq, vector.num_bits);
            for (dim, &value) in residual.iter().enumerate() {
                let ex_code = match vector.num_bits {
                    1 => 0,
                    num_bits => blocked_ex_code_value(&ex, dim, num_bits - 1),
                };
                builder.observe(dim, value, split_sign_bit(&signs, dim) as u8, ex_code);
            }
            let (got, want) = (builder.finish(), vector.factors());
            for (name, got, want) in [
                ("rho", got.rho, want.rho),
                ("gamma", got.gamma, want.gamma),
                ("unorm", got.unorm, want.unorm),
                ("ip_cent", got.ip_cent, want.ip_cent),
            ] {
                assert_close(got, want, &format!("{}.{name}", vector.id));
            }
        }
    }

    #[test]
    fn test_sym_dist_matches_lib() {
        let golden = golden();
        let vectors: HashMap<_, _> = golden
            .vectors
            .iter()
            .map(|vector| (vector.id.as_str(), vector))
            .collect();
        for pair in &golden.pairs {
            let (a, b) = (vectors[pair.a.as_str()], vectors[pair.b.as_str()]);
            let (signs_a, ex_a) = (bytes_from_hex(&a.bin_hex), bytes_from_hex(&a.ex_hex));
            let (signs_b, ex_b) = (bytes_from_hex(&b.bin_hex), bytes_from_hex(&b.ex_hex));
            let got = sym_dist(
                &SymCode {
                    signs: &signs_a,
                    ex: &ex_a,
                    factors: a.factors(),
                },
                &SymCode {
                    signs: &signs_b,
                    ex: &ex_b,
                    factors: b.factors(),
                },
                a.dim,
                a.num_bits,
                a.distance_type(),
            );
            let context = format!("{}-{}", pair.a, pair.b);
            assert_close(got, f32_from_hex(&pair.sym_dist), &context);
        }
    }

    /// Random codes in the production layout: sign words regrouped from
    /// `_rabit_codes` bytes, ex codes packed with zero padding.
    fn random_code(rng: &mut StdRng, dim: usize, num_bits: u8) -> (Vec<u8>, Vec<u8>) {
        let codes: Vec<u8> = (0..dim / 8).map(|_| rng.random()).collect();
        let signs = sign_words_from_codes(&codes, dim / 8);
        if num_bits == 1 {
            return (signs, Vec::new());
        }
        let ex_bits = num_bits - 1;
        let max_code = ((1u16 << ex_bits) - 1) as u8;
        let values: Vec<u8> = (0..dim).map(|_| rng.random_range(0..=max_code)).collect();
        let mut ex = vec![0; blocked_ex_code_bytes(dim, ex_bits)];
        pack_blocked_row(&values, ex_bits, &mut ex);
        (signs, ex)
    }

    #[test]
    fn test_centered_ip_simd_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(20260914);
        // 72/104/200 end in a zero-padded 64-dim block the SIMD kernels read.
        for dim in [8usize, 64, 72, 104, 128, 200, 576, 960] {
            for num_bits in 1u8..=9 {
                let (signs_a, ex_a) = random_code(&mut rng, dim, num_bits);
                let (signs_b, ex_b) = random_code(&mut rng, dim, num_bits);
                let dispatched = CENTERED_IP(&signs_a, &ex_a, &signs_b, &ex_b, dim, num_bits);
                let scalar =
                    centered_code_ip_scalar(&signs_a, &ex_a, &signs_b, &ex_b, dim, num_bits);
                assert_close(
                    dispatched,
                    scalar,
                    &format!("dim={dim} num_bits={num_bits}"),
                );
            }
        }
    }

    /// The warmup inner product of a row sums the 4-bit query codes over the
    /// dims whose `_rabit_codes` sign bit is set, in both query layouts.
    #[test]
    fn test_warmup_ip_sums_query_codes_over_set_signs() {
        let mut rng = StdRng::seed_from_u64(7);
        // Below 512 dims the query is block-major; 576/640/960 end in a
        // partial 512-dim superblock.
        for dim in [8usize, 64, 128, 512, 576, 640, 960] {
            let codes: Vec<u8> = (0..dim / 8).map(|_| rng.random()).collect();
            let signs = sign_words_from_codes(&codes, dim / 8);
            let query: Vec<u8> = (0..dim).map(|_| rng.random_range(0..16)).collect();
            let mut planes = vec![0u64; query_plane_words(dim)];
            transpose_query_for_warmup(&query, &mut planes);
            let is_set = |d: usize| (codes[d / 8] >> (d % 8)) & 1 == 1;
            let ip: u32 = (0..dim)
                .filter(|&d| is_set(d))
                .map(|d| query[d] as u32)
                .sum();
            let count = (0..dim).filter(|&d| is_set(d)).count();
            // `delta * ip + vl * popcount(signs)`
            assert_eq!(
                warmup_ip_x0_q(&signs, &planes, 1.0, 0.0),
                ip as f32,
                "{dim}"
            );
            assert_eq!(
                warmup_ip_x0_q_scalar(&signs, &planes, 1.0, 0.0),
                ip as f32,
                "{dim}"
            );
            assert_eq!(
                warmup_ip_x0_q(&signs, &planes, 0.0, 1.0),
                count as f32,
                "{dim}"
            );
        }
    }

    /// Lib `quantize_scalar`: a positive value sets bit 3 over its 3 extra
    /// bits, a non-positive one inverts the extra bits, and `vl` is `delta`
    /// times the code bias.
    #[test]
    fn test_quantize_scalar_matches_lib() {
        let zero = quantize_scalar_reconstruct(&[0.0; 64], 12.0);
        assert!(zero.codes.iter().all(|&code| code == 0));

        let mut query = vec![0.0f32; 64];
        query[0] = 1.0;
        query[1] = -1.0;
        let t_const = 14.0;
        let magnitude = 1.0 / 2.0f32.sqrt();
        let extra = (((t_const * magnitude as f64) + 1e-5) as i32).clamp(0, 7) as u8;
        let got = quantize_scalar_reconstruct(&query, t_const);
        assert_eq!(got.codes[0], extra + 8);
        assert_eq!(got.codes[1], !extra & 7);
        assert_eq!(got.codes[2], 7);
        assert!((got.vl - got.delta * -7.5).abs() <= 1e-6);
    }
}
