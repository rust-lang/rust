#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", debug_assertions))]
use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(all(target_arch = "aarch64", debug_assertions))]
static NEON_BACKEND_LOGGED: AtomicBool = AtomicBool::new(false);

#[cfg(all(target_arch = "aarch64", debug_assertions))]
fn log_backend_selection(_function_name: &str) {
    if !NEON_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
        // In a kernel environment, we might not have println!
        // This is just for sanity checking in debug mode
        #[cfg(feature = "std")]
        println!("SIMD backend selected for {}: NEON", _function_name);
    }
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
/// Input: t_lo and t_hi are uint16x8_t vectors (products).
/// Output: 8-bit results packed into a single uint8x16_t.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn apply_div255_neon(t_lo: uint16x8_t, t_hi: uint16x8_t) -> uint8x16_t {
    let one = vdupq_n_u16(1);

    // Apply (t + 1 + (t >> 8)) >> 8 to both halves
    let t_shr_lo = vshrq_n_u16(t_lo, 8);
    let t_shr_hi = vshrq_n_u16(t_hi, 8);

    let sum_lo = vaddq_u16(vaddq_u16(t_lo, one), t_shr_lo);
    let sum_hi = vaddq_u16(vaddq_u16(t_hi, one), t_shr_hi);

    let res_lo = vshrq_n_u16(sum_lo, 8);
    let res_hi = vshrq_n_u16(sum_hi, 8);

    // Narrow 16-bit to 8-bit
    let res_lo_u8 = vmovn_u16(res_lo);
    let res_hi_u8 = vmovn_u16(res_hi);

    vcombine_u8(res_lo_u8, res_hi_u8)
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
#[target_feature(enable = "neon")]
unsafe fn modulate_by_mask_neon(pixels: uint8x16_t, mask_vec: uint8x16_t) -> uint8x16_t {
    // Split into low and high halves for 16-bit operations
    let px_lo = vget_low_u8(pixels);
    let px_hi = vget_high_u8(pixels);
    let m_lo = vget_low_u8(mask_vec);
    let m_hi = vget_high_u8(mask_vec);

    // Widen to 16-bit for multiplication
    let px_lo_16 = vmovl_u8(px_lo);
    let px_hi_16 = vmovl_u8(px_hi);
    let m_lo_16 = vmovl_u8(m_lo);
    let m_hi_16 = vmovl_u8(m_hi);

    // Multiply channel * mask
    let t_lo = vmulq_u16(px_lo_16, m_lo_16);
    let t_hi = vmulq_u16(px_hi_16, m_hi_16);

    // Apply exact rounding: (t + 1 + (t >> 8)) >> 8
    apply_div255_neon(t_lo, t_hi)
}

/// Composite solid color with coverage mask (NEON backend).
///
/// Processes 4 pixels at a time using NEON for mask modulation, with scalar fallback
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
/// - NEON is available (function has `target_feature` annotation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn composite_solid_masked_over_neon(
    dst: &mut [u32],
    dst_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
    color_premul: u32,
) {
    #[cfg(debug_assertions)]
    log_backend_selection("composite_solid_masked_over");

    let ca = ((color_premul >> 24) & 0xFF) as u8;
    let cr = ((color_premul >> 16) & 0xFF) as u8;
    let cg = ((color_premul >> 8) & 0xFF) as u8;
    let cb = (color_premul & 0xFF) as u8;

    // Broadcast color to 4 pixels
    let color_px = vdupq_n_u32(color_premul);
    let color_u8 = vreinterpretq_u8_u32(color_px);

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

            // Pack masks into a vector: each mask byte repeated 4 times for RGBA
            // NEON byte order (little-endian): [m0, m0, m0, m0, m1, m1, m1, m1, ...]
            let mask_bytes = [
                m0 as u8, m0 as u8, m0 as u8, m0 as u8, m1 as u8, m1 as u8, m1 as u8, m1 as u8,
                m2 as u8, m2 as u8, m2 as u8, m2 as u8, m3 as u8, m3 as u8, m3 as u8, m3 as u8,
            ];
            let mask_vec = vld1q_u8(mask_bytes.as_ptr());

            // Modulate color by mask
            let src_modulated = modulate_by_mask_neon(color_u8, mask_vec);

            // Extract modulated pixels to array for scalar processing
            let mut src_array: [u32; 4] = [0; 4];
            vst1q_u8(src_array.as_mut_ptr() as *mut u8, src_modulated);

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

/// Composite source pixels with coverage mask (NEON backend).
///
/// Processes 4 pixels at a time using NEON for mask modulation, with scalar fallback
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
/// - NEON is available (function has `target_feature` annotation)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn composite_src_masked_over_neon(
    dst: &mut [u32],
    dst_stride: usize,
    src: &[u32],
    src_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
) {
    #[cfg(debug_assertions)]
    log_backend_selection("composite_src_masked_over");

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
            let src_ptr = src_row.as_ptr().add(x) as *const u8;
            let src_pixels = vld1q_u8(src_ptr);

            // Pack masks into a vector: each mask byte repeated 4 times for RGBA
            let mask_bytes = [
                m0 as u8, m0 as u8, m0 as u8, m0 as u8, m1 as u8, m1 as u8, m1 as u8, m1 as u8,
                m2 as u8, m2 as u8, m2 as u8, m2 as u8, m3 as u8, m3 as u8, m3 as u8, m3 as u8,
            ];
            let mask_vec = vld1q_u8(mask_bytes.as_ptr());

            // Modulate source by mask
            let src_modulated = modulate_by_mask_neon(src_pixels, mask_vec);

            // Extract modulated pixels to array for scalar processing
            let mut src_array: [u32; 4] = [0; 4];
            vst1q_u8(src_array.as_mut_ptr() as *mut u8, src_modulated);

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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn blit_rgba8888_over_neon(dst: &mut [u32], src: &[u32]) {
    let len = dst.len().min(src.len());
    let mut i = 0;

    while i + 4 <= len {
        let src_ptr = src.as_ptr().add(i);
        let dst_ptr = dst.as_mut_ptr().add(i);

        let s = vld1q_u32(src_ptr);
        let alphas = vshrq_n_u32(s, 24);

        if vmaxvq_u32(alphas) == 0 {
            i += 4;
            continue;
        }

        if vminvq_u32(alphas) == 255 {
            vst1q_u32(dst_ptr, s);
            i += 4;
            continue;
        }

        let d = vld1q_u32(dst_ptr);

        let a_step1 = vsliq_n_u32(alphas, alphas, 8);
        let a_step2 = vsliq_n_u32(a_step1, a_step1, 16);
        let a_u8 = vreinterpretq_u8_u32(a_step2);

        let v255 = vdupq_n_u8(255);
        let inv_a_u8 = vsubq_u8(v255, a_u8);

        let s_u8 = vreinterpretq_u8_u32(s);
        let d_u8 = vreinterpretq_u8_u32(d);

        let s_lo = vget_low_u8(s_u8);
        let s_hi = vget_high_u8(s_u8);
        let d_lo = vget_low_u8(d_u8);
        let d_hi = vget_high_u8(d_u8);

        let a_lo = vget_low_u8(a_u8);
        let a_hi = vget_high_u8(a_u8);
        let inv_a_lo = vget_low_u8(inv_a_u8);
        let inv_a_hi = vget_high_u8(inv_a_u8);

        let s_mul_lo = vmull_u8(s_lo, a_lo);
        let s_mul_hi = vmull_u8(s_hi, a_hi);
        let sum_lo = vmlal_u8(s_mul_lo, d_lo, inv_a_lo);
        let sum_hi = vmlal_u8(s_mul_hi, d_hi, inv_a_hi);

        let one = vdupq_n_u16(1);
        let t_shr_lo = vshrq_n_u16(sum_lo, 8);
        let t_shr_hi = vshrq_n_u16(sum_hi, 8);
        let res_lo_u16 = vshrq_n_u16(vaddq_u16(vaddq_u16(sum_lo, one), t_shr_lo), 8);
        let res_hi_u16 = vshrq_n_u16(vaddq_u16(vaddq_u16(sum_hi, one), t_shr_hi), 8);

        let res_lo_u8 = vmovn_u16(res_lo_u16);
        let res_hi_u8 = vmovn_u16(res_hi_u16);
        let result = vcombine_u8(res_lo_u8, res_hi_u8);
        let result_u32 = vreinterpretq_u32_u8(result);

        // Fix alpha per-lane to match scalar semantics
        let mut rgb_out: [u32; 4] = core::mem::transmute(result_u32);
        let dst_in: [u32; 4] = core::mem::transmute(d);
        let src_in: [u32; 4] = core::mem::transmute(s);
        for k in 0..4 {
            let s_px = src_in[k];
            let sa = (s_px >> 24) & 0xFF;
            if sa == 0 {
                continue;
            }
            if sa == 255 {
                rgb_out[k] = s_px;
                continue;
            }
            let d_px = dst_in[k];
            let da = (d_px >> 24) & 0xFF;
            let out_a = sa + scale_ch(da as u8, (255 - sa) as u8);
            rgb_out[k] = (out_a << 24) | (rgb_out[k] & 0x00FF_FFFF);
        }

        vst1q_u32(dst_ptr, core::mem::transmute(rgb_out));
        i += 4;
    }

    if i < len {
        crate::simd::scalar::blit_rgba8888_over_scalar(&mut dst[i..], &src[i..]);
    }
}
