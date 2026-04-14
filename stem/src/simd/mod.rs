//! Simple SIMD abstraction for pixel blits.
//! Provides scalar reference and architecture backends with identical math.
//! The scalar implementation is the canonical truth; SIMD backends must match it exactly.

#[cfg(target_arch = "aarch64")]
mod neon;
pub mod scalar;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
mod x86;

pub mod text;

/// Blend src over dst (premultiplied RGBA8888).
pub fn blit_rgba8888_over(dst: &mut [u32], src: &[u32]) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    #[cfg(target_feature = "sse2")]
    unsafe {
        x86::blit_rgba8888_over_sse2(dst, src);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    unsafe {
        neon::blit_rgba8888_over_neon(dst, src);
        return;
    }

    #[allow(unreachable_code)]
    scalar::blit_rgba8888_over_scalar(dst, src);
}

/// Composite solid color with coverage mask (premultiplied RGBA8888).
///
/// Applies an 8-bit coverage mask to a solid color and composites it over the destination.
/// - `mask=0`: no change
/// - `mask=255`: full color application
/// - intermediate: proportional blend
///
/// Uses premultiplied alpha math with exact rounding to match scalar reference.
pub fn composite_solid_masked_over(
    dst: &mut [u32],
    dst_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
    color_premul: u32,
) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    {
        // Prefer AVX2 if available at runtime
        if x86::is_avx2_available() {
            unsafe {
                x86::composite_solid_masked_over_avx2(
                    dst,
                    dst_stride,
                    mask,
                    mask_stride,
                    rect_w,
                    rect_h,
                    color_premul,
                );
                return;
            }
        }

        // Fall back to SSE2 if available
        #[cfg(target_feature = "sse2")]
        unsafe {
            x86::composite_solid_masked_over_sse2(
                dst,
                dst_stride,
                mask,
                mask_stride,
                rect_w,
                rect_h,
                color_premul,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    unsafe {
        neon::composite_solid_masked_over_neon(
            dst,
            dst_stride,
            mask,
            mask_stride,
            rect_w,
            rect_h,
            color_premul,
        );
        return;
    }

    #[allow(unreachable_code)]
    scalar::composite_solid_masked_over_scalar(
        dst,
        dst_stride,
        mask,
        mask_stride,
        rect_w,
        rect_h,
        color_premul,
    );
}

/// Composite source pixels with coverage mask (premultiplied RGBA8888).
///
/// Applies an 8-bit coverage mask to source pixels and composites them over the destination.
/// This is the canonical antialiased edge compositor.
///
/// Uses premultiplied alpha math with exact rounding to match scalar reference.
pub fn composite_src_masked_over(
    dst: &mut [u32],
    dst_stride: usize,
    src: &[u32],
    src_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    rect_w: usize,
    rect_h: usize,
) {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    {
        // Prefer AVX2 if available at runtime
        if x86::is_avx2_available() {
            unsafe {
                x86::composite_src_masked_over_avx2(
                    dst,
                    dst_stride,
                    src,
                    src_stride,
                    mask,
                    mask_stride,
                    rect_w,
                    rect_h,
                );
                return;
            }
        }

        // Fall back to SSE2 if available
        #[cfg(target_feature = "sse2")]
        unsafe {
            x86::composite_src_masked_over_sse2(
                dst,
                dst_stride,
                src,
                src_stride,
                mask,
                mask_stride,
                rect_w,
                rect_h,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    unsafe {
        neon::composite_src_masked_over_neon(
            dst,
            dst_stride,
            src,
            src_stride,
            mask,
            mask_stride,
            rect_w,
            rect_h,
        );
        return;
    }

    #[allow(unreachable_code)]
    scalar::composite_src_masked_over_scalar(
        dst,
        dst_stride,
        src,
        src_stride,
        mask,
        mask_stride,
        rect_w,
        rect_h,
    );
}

#[cfg(test)]
mod tests;
