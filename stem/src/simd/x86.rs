// x86 SSE2/AVX2 backend
// Provides runtime-dispatched SIMD implementations for x86/x86_64 architectures
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::sync::atomic::{AtomicU8, Ordering};

// Cached AVX2 availability: 0 = unknown, 1 = not available, 2 = available
static AVX2_AVAILABLE: AtomicU8 = AtomicU8::new(0);

// Runtime detection for AVX2 (cached)
#[cfg(target_arch = "x86")]
pub(crate) fn is_avx2_available() -> bool {
    let cached = AVX2_AVAILABLE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 2;
    }

    use core::arch::x86::__cpuid_count;
    let available = unsafe {
        let cpuid = __cpuid_count(7, 0);
        (cpuid.ebx & (1 << 5)) != 0
    };

    AVX2_AVAILABLE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
    available
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn is_avx2_available() -> bool {
    let cached = AVX2_AVAILABLE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 2;
    }

    use core::arch::x86_64::__cpuid_count;
    let available = {
        let cpuid = __cpuid_count(7, 0);
        (cpuid.ebx & (1 << 5)) != 0
    };

    AVX2_AVAILABLE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
    available
}

#[inline(always)]
fn scale_ch(c: u8, a: u8) -> u32 {
    let t = c as u32 * a as u32;
    (t + 1 + (t >> 8)) >> 8
}

#[inline(always)]
fn blend_channel(s: u32, d: u32, sa: u32) -> u32 {
    let inv = 255 - sa;
    let t = s * sa + d * inv;
    (t + 1 + (t >> 8)) >> 8
}

/// Apply the exact rounding formula: (t + 1 + (t >> 8)) >> 8 to 16-bit values.
/// Input: t_lo and t_hi are __m128i with 8x u16 values each (products).
/// Output: 8-bit results packed into a single __m128i (16 u8 values).
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn apply_div255_sse2(t_lo: __m128i, t_hi: __m128i, out: &mut __m128i) {
    let one = _mm_set1_epi16(1);

    // Apply (t + 1 + (t >> 8)) >> 8 to both halves
    let t_shr_lo = _mm_srli_epi16(t_lo, 8);
    let t_shr_hi = _mm_srli_epi16(t_hi, 8);

    let sum_lo = _mm_add_epi16(_mm_add_epi16(t_lo, one), t_shr_lo);
    let sum_hi = _mm_add_epi16(_mm_add_epi16(t_hi, one), t_shr_hi);

    let res_lo = _mm_srli_epi16(sum_lo, 8);
    let res_hi = _mm_srli_epi16(sum_hi, 8);

    *out = _mm_packus_epi16(res_lo, res_hi);
}

/// Modulate 4 RGBA pixels by a mask, returning modulated u8 channels.
///
/// # Layout
/// The mask_vec should contain mask values replicated across each pixel's 4 channels:
/// - Bytes 0-3: First pixel's mask repeated 4 times (M0, M0, M0, M0)
/// - Bytes 4-7: Second pixel's mask repeated 4 times (M1, M1, M1, M1)
/// - Bytes 8-11: Third pixel's mask repeated 4 times (M2, M2, M2, M2)
/// - Bytes 12-15: Fourth pixel's mask repeated 4 times (M3, M3, M3, M3)
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn modulate_by_mask_sse2(pixels: __m128i, mask_vec: __m128i, out: &mut __m128i) {
    let zero = _mm_setzero_si128();

    // Unpack pixels to 16-bit
    let px_lo = _mm_unpacklo_epi8(pixels, zero);
    let px_hi = _mm_unpackhi_epi8(pixels, zero);

    // Unpack mask to 16-bit
    let m_lo = _mm_unpacklo_epi8(mask_vec, zero);
    let m_hi = _mm_unpackhi_epi8(mask_vec, zero);

    // Multiply channel * mask
    let t_lo = _mm_mullo_epi16(px_lo, m_lo);
    let t_hi = _mm_mullo_epi16(px_hi, m_hi);

    // Apply exact rounding: (t + 1 + (t >> 8)) >> 8
    apply_div255_sse2(t_lo, t_hi, out);
}

/// Composite solid color with coverage mask (SSE2 backend).
///
/// Processes 4 pixels at a time using SSE2 for mask modulation, with scalar fallback
/// for tail pixels and over blend operation.
///
/// Math contract (canonical - matches scalar exactly):
/// - All inputs/outputs are premultiplied RGBA8888
/// - Coverage mask modulates the color's alpha and RGB channels
/// - Modulation uses: `result = (channel * mask + 1 + (channel * mask >> 8)) >> 8`
///   This is a fast approximation of `(channel * mask) / 255` with exact rounding
/// - After modulation, applies over operator: `dst = color' + dst * (1 - color_a')`
/// - Over blend also uses `(t + 1 + (t >> 8)) >> 8` rounding
/// - Mask values: 0 = no change, 255 = full color, intermediate = proportional blend
///
/// # Safety
/// Caller must ensure:
/// - `dst.len() >= dst_stride * rect_h`
/// - `mask.len() >= mask_stride * rect_h`
/// - `dst_stride >= rect_w` and `mask_stride >= rect_w`
/// - SSE2 is available (function has `target_feature` annotation)
#[target_feature(enable = "sse2")]
pub unsafe fn composite_solid_masked_over_sse2(
    dst: &mut [u32],
    dst_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
    color_premul: u32,
) {
    let ca = ((color_premul >> 24) & 0xFF) as u8;
    let cr = ((color_premul >> 16) & 0xFF) as u8;
    let cg = ((color_premul >> 8) & 0xFF) as u8;
    let cb = (color_premul & 0xFF) as u8;

    // Broadcast color to 4 pixels
    let color_px = _mm_set1_epi32(color_premul as i32);

    for y in 0..rect_h {
        let dst_row = &mut dst[y * dst_stride..];
        let mask_row = &mask[y * mask_stride..];

        let mut x = 0;

        // Process 4 pixels at a time
        while x + 4 <= rect_w {
            // Load 4 mask values
            let m0 = mask_row[x] as u32;
            let m1 = mask_row[x + 1] as u32;
            let m2 = mask_row[x + 2] as u32;
            let m3 = mask_row[x + 3] as u32;

            // Check if all masks are zero (early out)
            if (m0 | m1 | m2 | m3) == 0 {
                x += 4;
                continue;
            }

            // Pack masks into an __m128i: each mask byte repeated 4 times for RGBA
            let mask_vec = _mm_set_epi8(
                m3 as i8, m3 as i8, m3 as i8, m3 as i8, m2 as i8, m2 as i8, m2 as i8, m2 as i8,
                m1 as i8, m1 as i8, m1 as i8, m1 as i8, m0 as i8, m0 as i8, m0 as i8, m0 as i8,
            );

            // Modulate color by mask
            let mut src_modulated = _mm_setzero_si128();
            modulate_by_mask_sse2(color_px, mask_vec, &mut src_modulated);

            // Extract modulated pixels to array for scalar processing
            let mut src_array: [u32; 4] = [0; 4];
            _mm_storeu_si128(src_array.as_mut_ptr() as *mut __m128i, src_modulated);

            for i in 0..4 {
                let src_px = src_array[i];
                let sa = (src_px >> 24) & 0xFF;

                if sa == 0 {
                    continue;
                }

                if sa == 255 {
                    dst_row[x + i] = src_px;
                    continue;
                }

                let sr = (src_px >> 16) & 0xFF;
                let sg = (src_px >> 8) & 0xFF;
                let sb = src_px & 0xFF;

                let dv = dst_row[x + i];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x + i] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 4;
        }

        // Handle remaining pixels with scalar
        while x < rect_w {
            let m = mask_row[x];
            if m == 0 {
                x += 1;
                continue;
            }

            let sa = scale_ch(ca, m);
            let sr = scale_ch(cr, m);
            let sg = scale_ch(cg, m);
            let sb = scale_ch(cb, m);

            if sa == 255 {
                dst_row[x] = (sa << 24) | (sr << 16) | (sg << 8) | sb;
            } else if sa > 0 {
                let dv = dst_row[x];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 1;
        }
    }
}

/// Composite source pixels with coverage mask (SSE2 backend).
///
/// Processes 4 pixels at a time using SSE2 for mask modulation, with scalar fallback
/// for tail pixels and over blend operation.
///
/// Math contract (canonical - matches scalar exactly):
/// - All inputs/outputs are premultiplied RGBA8888
/// - Mask modulates source alpha and RGB channels
/// - Modulation uses: `result = (channel * mask + 1 + (channel * mask >> 8)) >> 8`
///   This is a fast approximation of `(channel * mask) / 255` with exact rounding
/// - After modulation, applies over operator: `dst = src' + dst * (1 - src_a')`
/// - Over blend also uses `(t + 1 + (t >> 8)) >> 8` rounding
///
/// # Safety
/// Caller must ensure:
/// - `dst.len() >= dst_stride * rect_h`
/// - `src.len() >= src_stride * rect_h`
/// - `mask.len() >= mask_stride * rect_h`
/// - `dst_stride >= rect_w`, `src_stride >= rect_w`, `mask_stride >= rect_w`
/// - SSE2 is available (function has `target_feature` annotation)
#[target_feature(enable = "sse2")]
pub unsafe fn composite_src_masked_over_sse2(
    dst: &mut [u32],
    dst_stride: usize,
    src: &[u32],
    src_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
) {
    for y in 0..rect_h {
        let dst_row = &mut dst[y * dst_stride..];
        let src_row = &src[y * src_stride..];
        let mask_row = &mask[y * mask_stride..];

        let mut x = 0;

        // Process 4 pixels at a time
        while x + 4 <= rect_w {
            // Load 4 mask values
            let m0 = mask_row[x] as u32;
            let m1 = mask_row[x + 1] as u32;
            let m2 = mask_row[x + 2] as u32;
            let m3 = mask_row[x + 3] as u32;

            // Check if all masks are zero (early out)
            if (m0 | m1 | m2 | m3) == 0 {
                x += 4;
                continue;
            }

            // Load 4 source pixels
            let src_ptr = src_row.as_ptr().add(x) as *const __m128i;
            let src_pixels = _mm_loadu_si128(src_ptr);

            // Pack masks into an __m128i: each mask byte repeated 4 times for RGBA
            let mask_vec = _mm_set_epi8(
                m3 as i8, m3 as i8, m3 as i8, m3 as i8, m2 as i8, m2 as i8, m2 as i8, m2 as i8,
                m1 as i8, m1 as i8, m1 as i8, m1 as i8, m0 as i8, m0 as i8, m0 as i8, m0 as i8,
            );

            // Modulate source by mask
            let mut src_modulated = _mm_setzero_si128();
            modulate_by_mask_sse2(src_pixels, mask_vec, &mut src_modulated);

            // For now, fall back to scalar per-pixel for the over blend
            // This ensures bit-exact results
            let mut src_array: [u32; 4] = [0; 4];
            _mm_storeu_si128(src_array.as_mut_ptr() as *mut __m128i, src_modulated);

            for i in 0..4 {
                let src_px = src_array[i];
                let sa = (src_px >> 24) & 0xFF;

                if sa == 0 {
                    continue;
                }

                if sa == 255 {
                    dst_row[x + i] = src_px;
                    continue;
                }

                let sr = (src_px >> 16) & 0xFF;
                let sg = (src_px >> 8) & 0xFF;
                let sb = src_px & 0xFF;

                let dv = dst_row[x + i];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x + i] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 4;
        }

        // Handle remaining pixels with scalar
        while x < rect_w {
            let m = mask_row[x];
            if m == 0 {
                x += 1;
                continue;
            }

            let s = src_row[x];
            let sa_orig = ((s >> 24) & 0xFF) as u8;
            let sr_orig = ((s >> 16) & 0xFF) as u8;
            let sg_orig = ((s >> 8) & 0xFF) as u8;
            let sb_orig = (s & 0xFF) as u8;

            let sa = scale_ch(sa_orig, m);
            let sr = scale_ch(sr_orig, m);
            let sg = scale_ch(sg_orig, m);
            let sb = scale_ch(sb_orig, m);

            if sa == 255 {
                dst_row[x] = (sa << 24) | (sr << 16) | (sg << 8) | sb;
            } else if sa > 0 {
                let dv = dst_row[x];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 1;
        }
    }
}

#[target_feature(enable = "sse2")]
pub unsafe fn blit_rgba8888_over_sse2(dst: &mut [u32], src: &[u32]) {
    let len = dst.len().min(src.len());
    let mut i = 0;

    while i + 4 <= len {
        let src_ptr = src.as_ptr().add(i) as *const __m128i;
        let dst_ptr = dst.as_mut_ptr().add(i) as *mut __m128i;

        let s = _mm_loadu_si128(src_ptr);
        let alphas = _mm_srli_epi32(s, 24);

        let zero = _mm_setzero_si128();
        let cmp_zero = _mm_cmpeq_epi32(alphas, zero);
        if _mm_movemask_epi8(cmp_zero) == 0xFFFF {
            i += 4;
            continue;
        }

        let alpha_255 = _mm_set1_epi32(255);
        let cmp_255 = _mm_cmpeq_epi32(alphas, alpha_255);
        if _mm_movemask_epi8(cmp_255) == 0xFFFF {
            _mm_storeu_si128(dst_ptr, s);
            i += 4;
            continue;
        }

        let d = _mm_loadu_si128(dst_ptr);

        let s_lo = _mm_unpacklo_epi8(s, zero);
        let s_hi = _mm_unpackhi_epi8(s, zero);
        let d_lo = _mm_unpacklo_epi8(d, zero);
        let d_hi = _mm_unpackhi_epi8(d, zero);

        let t1 = _mm_slli_epi32(alphas, 8);
        let t2 = _mm_or_si128(alphas, t1);
        let t3 = _mm_slli_epi32(t2, 16);
        let a_vec = _mm_or_si128(t2, t3);

        let a_lo = _mm_unpacklo_epi8(a_vec, zero);
        let a_hi = _mm_unpackhi_epi8(a_vec, zero);

        let v255 = _mm_set1_epi16(255);
        let inv_a_lo = _mm_sub_epi16(v255, a_lo);
        let inv_a_hi = _mm_sub_epi16(v255, a_hi);

        let s_mul_lo = _mm_mullo_epi16(s_lo, a_lo);
        let s_mul_hi = _mm_mullo_epi16(s_hi, a_hi);
        let d_mul_lo = _mm_mullo_epi16(d_lo, inv_a_lo);
        let d_mul_hi = _mm_mullo_epi16(d_hi, inv_a_hi);

        let sum_lo = _mm_add_epi16(s_mul_lo, d_mul_lo);
        let sum_hi = _mm_add_epi16(s_mul_hi, d_mul_hi);

        let one = _mm_set1_epi16(1);
        let t_shr_lo = _mm_srli_epi16(sum_lo, 8);
        let t_shr_hi = _mm_srli_epi16(sum_hi, 8);
        let t_lo = _mm_add_epi16(_mm_add_epi16(sum_lo, one), t_shr_lo);
        let t_hi = _mm_add_epi16(_mm_add_epi16(sum_hi, one), t_shr_hi);
        let res_lo = _mm_srli_epi16(t_lo, 8);
        let res_hi = _mm_srli_epi16(t_hi, 8);

        let result = _mm_packus_epi16(res_lo, res_hi);

        #[repr(align(16))]
        struct Buf([u32; 4]);
        let mut rgb_out = Buf([0; 4]);
        let mut dst_in = Buf([0; 4]);
        _mm_storeu_si128(rgb_out.0.as_mut_ptr() as *mut __m128i, result);
        _mm_storeu_si128(dst_in.0.as_mut_ptr() as *mut __m128i, d);
        let src_ptr_u32 = src.as_ptr().add(i);

        for k in 0..4 {
            let s_px = *src_ptr_u32.add(k);
            let sa = (s_px >> 24) & 0xFF;
            if sa == 0 {
                continue;
            }
            if sa == 255 {
                rgb_out.0[k] = s_px;
                continue;
            }
            let d_px = dst_in.0[k];
            let da = (d_px >> 24) & 0xFF;
            let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
            rgb_out.0[k] = (out_a << 24) | (rgb_out.0[k] & 0x00FF_FFFF);
        }

        _mm_storeu_si128(dst_ptr, core::mem::transmute(rgb_out.0));
        i += 4;
    }

    if i < len {
        crate::simd::scalar::blit_rgba8888_over_scalar(&mut dst[i..], &src[i..]);
    }
}

// ============================================================================
// AVX2 Backend (256-bit, 8 pixels at a time)
// ============================================================================

/// Apply the exact rounding formula: (t + 1 + (t >> 8)) >> 8 to 16-bit values (AVX2 version).
/// Input: t_lo and t_hi are __m256i with 16x u16 values each (products).
/// Output: 8-bit results packed into a single __m256i (32 u8 values).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn apply_div255_avx2(t_lo: __m256i, t_hi: __m256i, out: &mut __m256i) {
    let one = _mm256_set1_epi16(1);

    // Apply (t + 1 + (t >> 8)) >> 8 to both halves
    let t_shr_lo = _mm256_srli_epi16(t_lo, 8);
    let t_shr_hi = _mm256_srli_epi16(t_hi, 8);

    let sum_lo = _mm256_add_epi16(_mm256_add_epi16(t_lo, one), t_shr_lo);
    let sum_hi = _mm256_add_epi16(_mm256_add_epi16(t_hi, one), t_shr_hi);

    let res_lo = _mm256_srli_epi16(sum_lo, 8);
    let res_hi = _mm256_srli_epi16(sum_hi, 8);

    *out = _mm256_packus_epi16(res_lo, res_hi);
}

/// Modulate 8 RGBA pixels by a mask, returning modulated u8 channels (AVX2 version).
///
/// # Layout
/// The mask_vec should contain mask values replicated across each pixel's 4 channels:
/// - Each 4-byte group contains one mask repeated 4 times (M, M, M, M)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn modulate_by_mask_avx2(pixels: __m256i, mask_vec: __m256i, out: &mut __m256i) {
    let zero = _mm256_setzero_si256();

    // Unpack pixels to 16-bit
    let px_lo = _mm256_unpacklo_epi8(pixels, zero);
    let px_hi = _mm256_unpackhi_epi8(pixels, zero);

    // Unpack mask to 16-bit
    let m_lo = _mm256_unpacklo_epi8(mask_vec, zero);
    let m_hi = _mm256_unpackhi_epi8(mask_vec, zero);

    // Multiply channel * mask
    let t_lo = _mm256_mullo_epi16(px_lo, m_lo);
    let t_hi = _mm256_mullo_epi16(px_hi, m_hi);

    // Apply exact rounding: (t + 1 + (t >> 8)) >> 8
    apply_div255_avx2(t_lo, t_hi, out);
}

/// Composite solid color with coverage mask (AVX2 backend).
///
/// Processes 8 pixels at a time using AVX2 for mask modulation, with scalar fallback
/// for tail pixels and over blend operation.
///
/// Math contract (canonical - matches scalar exactly):
/// - Same as SSE2 version, but processes 8 pixels at once
///
/// # Safety
/// Caller must ensure:
/// - `dst.len() >= dst_stride * rect_h`
/// - `mask.len() >= mask_stride * rect_h`
/// - `dst_stride >= rect_w` and `mask_stride >= rect_w`
/// - AVX2 is available (function has `target_feature` annotation)
#[target_feature(enable = "avx2")]
pub unsafe fn composite_solid_masked_over_avx2(
    dst: &mut [u32],
    dst_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
    color_premul: u32,
) {
    let ca = ((color_premul >> 24) & 0xFF) as u8;
    let cr = ((color_premul >> 16) & 0xFF) as u8;
    let cg = ((color_premul >> 8) & 0xFF) as u8;
    let cb = (color_premul & 0xFF) as u8;

    // Broadcast color to 8 pixels
    let color_px = _mm256_set1_epi32(color_premul as i32);

    for y in 0..rect_h {
        let dst_row = &mut dst[y * dst_stride..];
        let mask_row = &mask[y * mask_stride..];

        let mut x = 0;

        // Process 8 pixels at a time with AVX2
        while x + 8 <= rect_w {
            // Load 8 mask values
            let masks = [
                mask_row[x] as u32,
                mask_row[x + 1] as u32,
                mask_row[x + 2] as u32,
                mask_row[x + 3] as u32,
                mask_row[x + 4] as u32,
                mask_row[x + 5] as u32,
                mask_row[x + 6] as u32,
                mask_row[x + 7] as u32,
            ];

            // Check if all masks are zero (early out)
            if masks.iter().all(|&m| m == 0) {
                x += 8;
                continue;
            }

            // Pack masks into an __m256i: each mask byte repeated 4 times for RGBA
            let mask_vec = _mm256_set_epi8(
                masks[7] as i8,
                masks[7] as i8,
                masks[7] as i8,
                masks[7] as i8,
                masks[6] as i8,
                masks[6] as i8,
                masks[6] as i8,
                masks[6] as i8,
                masks[5] as i8,
                masks[5] as i8,
                masks[5] as i8,
                masks[5] as i8,
                masks[4] as i8,
                masks[4] as i8,
                masks[4] as i8,
                masks[4] as i8,
                masks[3] as i8,
                masks[3] as i8,
                masks[3] as i8,
                masks[3] as i8,
                masks[2] as i8,
                masks[2] as i8,
                masks[2] as i8,
                masks[2] as i8,
                masks[1] as i8,
                masks[1] as i8,
                masks[1] as i8,
                masks[1] as i8,
                masks[0] as i8,
                masks[0] as i8,
                masks[0] as i8,
                masks[0] as i8,
            );

            // Modulate color by mask
            let mut src_modulated = _mm256_setzero_si256();
            modulate_by_mask_avx2(color_px, mask_vec, &mut src_modulated);

            // Extract modulated pixels to array for scalar processing
            let mut src_array: [u32; 8] = [0; 8];
            _mm256_storeu_si256(src_array.as_mut_ptr() as *mut __m256i, src_modulated);

            for i in 0..8 {
                let src_px = src_array[i];
                let sa = (src_px >> 24) & 0xFF;

                if sa == 0 {
                    continue;
                }

                if sa == 255 {
                    dst_row[x + i] = src_px;
                    continue;
                }

                let sr = (src_px >> 16) & 0xFF;
                let sg = (src_px >> 8) & 0xFF;
                let sb = src_px & 0xFF;

                let dv = dst_row[x + i];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x + i] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 8;
        }

        // Handle remaining pixels with scalar
        while x < rect_w {
            let m = mask_row[x];
            if m == 0 {
                x += 1;
                continue;
            }

            let sa = scale_ch(ca, m);
            let sr = scale_ch(cr, m);
            let sg = scale_ch(cg, m);
            let sb = scale_ch(cb, m);

            if sa == 255 {
                dst_row[x] = (sa << 24) | (sr << 16) | (sg << 8) | sb;
            } else if sa > 0 {
                let dv = dst_row[x];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 1;
        }
    }
}

/// Composite source pixels with coverage mask (AVX2 backend).
///
/// Processes 8 pixels at a time using AVX2 for mask modulation, with scalar fallback
/// for tail pixels and over blend operation.
///
/// Math contract (canonical - matches scalar exactly):
/// - Same as SSE2 version, but processes 8 pixels at once
///
/// # Safety
/// Caller must ensure:
/// - `dst.len() >= dst_stride * rect_h`
/// - `src.len() >= src_stride * rect_h`
/// - `mask.len() >= mask_stride * rect_h`
/// - `dst_stride >= rect_w`, `src_stride >= rect_w`, `mask_stride >= rect_w`
/// - AVX2 is available (function has `target_feature` annotation)
#[target_feature(enable = "avx2")]
pub unsafe fn composite_src_masked_over_avx2(
    dst: &mut [u32],
    dst_stride: usize,
    src: &[u32],
    src_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
) {
    for y in 0..rect_h {
        let dst_row = &mut dst[y * dst_stride..];
        let src_row = &src[y * src_stride..];
        let mask_row = &mask[y * mask_stride..];

        let mut x = 0;

        // Process 8 pixels at a time with AVX2
        while x + 8 <= rect_w {
            // Load 8 mask values
            let masks = [
                mask_row[x] as u32,
                mask_row[x + 1] as u32,
                mask_row[x + 2] as u32,
                mask_row[x + 3] as u32,
                mask_row[x + 4] as u32,
                mask_row[x + 5] as u32,
                mask_row[x + 6] as u32,
                mask_row[x + 7] as u32,
            ];

            // Check if all masks are zero (early out)
            if masks.iter().all(|&m| m == 0) {
                x += 8;
                continue;
            }

            // Load 8 source pixels
            let src_ptr = src_row.as_ptr().add(x) as *const __m256i;
            let src_pixels = _mm256_loadu_si256(src_ptr);

            // Pack masks into an __m256i: each mask byte repeated 4 times for RGBA
            let mask_vec = _mm256_set_epi8(
                masks[7] as i8,
                masks[7] as i8,
                masks[7] as i8,
                masks[7] as i8,
                masks[6] as i8,
                masks[6] as i8,
                masks[6] as i8,
                masks[6] as i8,
                masks[5] as i8,
                masks[5] as i8,
                masks[5] as i8,
                masks[5] as i8,
                masks[4] as i8,
                masks[4] as i8,
                masks[4] as i8,
                masks[4] as i8,
                masks[3] as i8,
                masks[3] as i8,
                masks[3] as i8,
                masks[3] as i8,
                masks[2] as i8,
                masks[2] as i8,
                masks[2] as i8,
                masks[2] as i8,
                masks[1] as i8,
                masks[1] as i8,
                masks[1] as i8,
                masks[1] as i8,
                masks[0] as i8,
                masks[0] as i8,
                masks[0] as i8,
                masks[0] as i8,
            );

            // Modulate source by mask
            let mut src_modulated = _mm256_setzero_si256();
            modulate_by_mask_avx2(src_pixels, mask_vec, &mut src_modulated);

            // For now, fall back to scalar per-pixel for the over blend
            // This ensures bit-exact results
            let mut src_array: [u32; 8] = [0; 8];
            _mm256_storeu_si256(src_array.as_mut_ptr() as *mut __m256i, src_modulated);

            for i in 0..8 {
                let src_px = src_array[i];
                let sa = (src_px >> 24) & 0xFF;

                if sa == 0 {
                    continue;
                }

                if sa == 255 {
                    dst_row[x + i] = src_px;
                    continue;
                }

                let sr = (src_px >> 16) & 0xFF;
                let sg = (src_px >> 8) & 0xFF;
                let sb = src_px & 0xFF;

                let dv = dst_row[x + i];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x + i] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 8;
        }

        // Handle remaining pixels with scalar
        while x < rect_w {
            let m = mask_row[x];
            if m == 0 {
                x += 1;
                continue;
            }

            let s = src_row[x];
            let sa_orig = ((s >> 24) & 0xFF) as u8;
            let sr_orig = ((s >> 16) & 0xFF) as u8;
            let sg_orig = ((s >> 8) & 0xFF) as u8;
            let sb_orig = (s & 0xFF) as u8;

            let sa = scale_ch(sa_orig, m);
            let sr = scale_ch(sr_orig, m);
            let sg = scale_ch(sg_orig, m);
            let sb = scale_ch(sb_orig, m);

            if sa == 255 {
                dst_row[x] = (sa << 24) | (sr << 16) | (sg << 8) | sb;
            } else if sa > 0 {
                let dv = dst_row[x];
                let da = (dv >> 24) & 0xFF;
                let dr = (dv >> 16) & 0xFF;
                let dg = (dv >> 8) & 0xFF;
                let db = dv & 0xFF;

                let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
                let out_r = blend_channel(sr, dr, sa);
                let out_g = blend_channel(sg, dg, sa);
                let out_b = blend_channel(sb, db, sa);

                dst_row[x] = (out_a << 24) | (out_r << 16) | (out_g << 8) | out_b;
            }

            x += 1;
        }
    }
}
