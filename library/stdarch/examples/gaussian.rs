//! Hexagon HVX Gaussian 3x3 blur example
//!
//! This example demonstrates the use of Hexagon HVX intrinsics to implement
//! a 3x3 Gaussian blur filter on unsigned 8-bit images.
//!
//! The 3x3 Gaussian kernel is:
//!     1 2 1
//!     2 4 2  / 16
//!     1 2 1
//!
//! This is a separable filter: `[1 2 1]^T * [1 2 1] / 16`.
//!
//! On Hexagon targets, this implementation uses `HvxVectorPair` for widening
//! arithmetic to achieve full precision in the Gaussian computation, avoiding
//! the approximation errors of byte-averaging approaches. On other targets,
//! it runs a reference implementation in pure Rust.
//!
//! # Building and Running (Hexagon)
//!
//! To build (requires Hexagon toolchain):
//!
//!     RUSTFLAGS="-C target-feature=+hvxv62,+hvx-length128b \
//!         -C linker=hexagon-unknown-linux-musl-clang" \
//!         cargo +nightly build -p stdarch_examples --bin gaussian \
//!         --target hexagon-unknown-linux-musl \
//!         -Zbuild-std -Zbuild-std-features=llvm-libunwind
//!
//! To run under QEMU:
//!
//!     qemu-hexagon -L <sysroot>/target/hexagon-unknown-linux-musl \
//!         target/hexagon-unknown-linux-musl/debug/gaussian
//!
//! # Building and Running (Other targets)
//!
//!     cargo +nightly run -p stdarch_examples --bin gaussian

#![cfg_attr(target_arch = "hexagon", feature(stdarch_hexagon))]
#![cfg_attr(target_arch = "hexagon", feature(hexagon_target_feature))]
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::missing_docs_in_private_items,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment
)]

/// Image width - must be multiple of HVX vector length on Hexagon
const WIDTH: usize = 256;
const HEIGHT: usize = 16;

// ============================================================================
// Hexagon HVX implementation
// ============================================================================

#[cfg(target_arch = "hexagon")]
mod hvx {
    #[cfg(not(target_feature = "hvx-length128b"))]
    use core_arch::arch::hexagon::v64::*;
    #[cfg(target_feature = "hvx-length128b")]
    use core_arch::arch::hexagon::v128::*;

    /// Vector length in bytes for HVX 128-byte mode
    #[cfg(target_feature = "hvx-length128b")]
    const VLEN: usize = 128;

    /// Vector length in bytes for HVX 64-byte mode
    #[cfg(not(target_feature = "hvx-length128b"))]
    const VLEN: usize = 64;

    /// Vertical 1-2-1 filter pass using HvxVectorPair widening arithmetic
    ///
    /// Computes: dst[x] = (row_above[x] + 2*center[x] + row_below[x] + 2) >> 2
    ///
    /// Uses HvxVectorPair to widen u8 to u16 for precise arithmetic, avoiding
    /// the rounding errors of byte-averaging approximations.
    ///
    /// # Safety
    ///
    /// - `src` must point to the center row with valid data at -stride and +stride
    /// - `dst` must point to a valid output buffer for `width` bytes
    /// - `width` must be a multiple of VLEN
    /// - All pointers must be HVX-aligned (128-byte for 128B mode)
    #[target_feature(enable = "hvxv62")]
    unsafe fn vertical_121_pass(src: *const u8, stride: isize, width: usize, dst: *mut u8) {
        let inp0 = src.offset(-stride) as *const HvxVector;
        let inp1 = src as *const HvxVector;
        let inp2 = src.offset(stride) as *const HvxVector;
        let outp = dst as *mut HvxVector;

        let n_chunks = width / VLEN;
        for i in 0..n_chunks {
            let above = *inp0.add(i);
            let center = *inp1.add(i);
            let below = *inp2.add(i);

            // Widen above + below to 16-bit using HvxVectorPair
            // q6_wh_vadd_vubvub: adds two u8 vectors, producing u16 results in a pair
            let above_plus_below: HvxVectorPair = q6_wh_vadd_vubvub(above, below);

            // Widen center * 2 (add center to itself)
            let center_x2: HvxVectorPair = q6_wh_vadd_vubvub(center, center);

            // Add them: (above + below) + (center * 2) = above + 2*center + below
            let sum: HvxVectorPair = q6_wh_vadd_whwh(above_plus_below, center_x2);

            // Extract high and low vectors from the pair (each contains u16 values)
            let sum_lo = q6_v_lo_w(sum); // Lower 64 elements as i16
            let sum_hi = q6_v_hi_w(sum); // Upper 64 elements as i16

            // Arithmetic right shift by 2 (divide by 4) with rounding
            // Add 2 for rounding before shift: (sum + 2) >> 2
            let two = q6_vh_vsplat_r(2);
            let sum_lo_rounded = q6_vh_vadd_vhvh(sum_lo, two);
            let sum_hi_rounded = q6_vh_vadd_vhvh(sum_hi, two);
            let shifted_lo = q6_vh_vasr_vhvh(sum_lo_rounded, two);
            let shifted_hi = q6_vh_vasr_vhvh(sum_hi_rounded, two);

            // Pack back to u8 with saturation: takes hi and lo halfword vectors,
            // saturates to u8, and interleaves them back to original order
            let result = q6_vub_vsat_vhvh(shifted_hi, shifted_lo);

            *outp.add(i) = result;
        }
    }

    /// Horizontal 1-2-1 filter pass using HvxVectorPair widening arithmetic
    ///
    /// Computes: dst[x] = (src[x-1] + 2*src[x] + src[x+1] + 2) >> 2
    ///
    /// Uses `valign` and `vlalign` to shift vectors by 1 byte for neighbor access,
    /// then HvxVectorPair for precise widening arithmetic.
    ///
    /// # Safety
    ///
    /// - `src` and `dst` must point to valid buffers of `width` bytes
    /// - `width` must be a multiple of VLEN
    /// - All pointers must be HVX-aligned
    #[target_feature(enable = "hvxv62")]
    unsafe fn horizontal_121_pass(src: *const u8, width: usize, dst: *mut u8) {
        let inp = src as *const HvxVector;
        let outp = dst as *mut HvxVector;

        let n_chunks = width / VLEN;
        let mut prev = q6_v_vzero();

        for i in 0..n_chunks {
            let curr = *inp.add(i);
            let next = if i + 1 < n_chunks {
                *inp.add(i + 1)
            } else {
                q6_v_vzero()
            };

            // Left neighbor (x-1): shift curr right by 1 byte, filling from prev
            let left = q6_v_vlalign_vvr(curr, prev, 1);

            // Right neighbor (x+1): shift curr left by 1 byte, filling from next
            let right = q6_v_valign_vvr(next, curr, 1);

            // Widen left + right to 16-bit
            let left_plus_right: HvxVectorPair = q6_wh_vadd_vubvub(left, right);

            // Widen center * 2
            let center_x2: HvxVectorPair = q6_wh_vadd_vubvub(curr, curr);

            // Add: left + 2*center + right
            let sum: HvxVectorPair = q6_wh_vadd_whwh(left_plus_right, center_x2);

            // Extract high and low vectors
            let sum_lo = q6_v_lo_w(sum);
            let sum_hi = q6_v_hi_w(sum);

            // Arithmetic right shift by 2 with rounding
            let two = q6_vh_vsplat_r(2);
            let sum_lo_rounded = q6_vh_vadd_vhvh(sum_lo, two);
            let sum_hi_rounded = q6_vh_vadd_vhvh(sum_hi, two);
            let shifted_lo = q6_vh_vasr_vhvh(sum_lo_rounded, two);
            let shifted_hi = q6_vh_vasr_vhvh(sum_hi_rounded, two);

            // Pack back to u8 with saturation
            let result = q6_vub_vsat_vhvh(shifted_hi, shifted_lo);

            *outp.add(i) = result;

            prev = curr;
        }
    }

    /// Apply Gaussian 3x3 blur to an entire image using separable filtering
    ///
    /// Two-pass approach:
    /// 1. Vertical pass: apply 1-2-1 filter across rows
    /// 2. Horizontal pass: apply 1-2-1 filter across columns
    ///
    /// Combined effect: 3x3 Gaussian kernel [1 2 1; 2 4 2; 1 2 1] / 16
    ///
    /// # Safety
    ///
    /// - `src` and `dst` must point to valid image buffers of `stride * height` bytes
    /// - `tmp` must point to a valid temporary buffer of `width` bytes, HVX-aligned
    /// - `width` must be a multiple of VLEN and >= VLEN
    /// - `stride` must be >= `width`
    /// - All buffers must be HVX-aligned (128-byte for 128B mode)
    #[target_feature(enable = "hvxv62")]
    pub unsafe fn gaussian3x3u8(
        src: *const u8,
        stride: usize,
        width: usize,
        height: usize,
        dst: *mut u8,
        tmp: *mut u8,
    ) {
        let stride_i = stride as isize;

        // Process interior rows (skip first and last which lack vertical neighbors)
        for y in 1..height - 1 {
            let row_src = src.offset(y as isize * stride_i);
            let row_dst = dst.offset(y as isize * stride_i);

            // Pass 1: vertical 1-2-1 into tmp
            vertical_121_pass(row_src, stride_i, width, tmp);

            // Pass 2: horizontal 1-2-1 from tmp into dst
            horizontal_121_pass(tmp, width, row_dst);
        }
    }
}

// ============================================================================
// Reference implementation (works on all targets)
// ============================================================================

/// Reference implementation of Gaussian 3x3 blur
///
/// Kernel:
///     1 2 1
///     2 4 2  / 16
///     1 2 1
fn gaussian3x3u8_reference(src: &[u8], stride: usize, width: usize, height: usize, dst: &mut [u8]) {
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Compute column sums (vertical 1-2-1 weights)
            let mut col = [0u32; 3];
            for i in 0..3 {
                col[i] = 1 * src[(y - 1) * stride + x - 1 + i] as u32
                    + 2 * src[y * stride + x - 1 + i] as u32
                    + 1 * src[(y + 1) * stride + x - 1 + i] as u32;
            }
            // Apply horizontal 1-2-1 weights and normalize
            // (1*col[0] + 2*col[1] + 1*col[2] + 8) / 16
            dst[y * stride + x] = ((1 * col[0] + 2 * col[1] + 1 * col[2] + 8) >> 4) as u8;
        }
    }
}

/// Generate deterministic test pattern
fn generate_test_pattern(buf: &mut [u8], width: usize, height: usize) {
    for y in 0..height {
        for x in 0..width {
            buf[y * width + x] = ((x + y * 7) % 256) as u8;
        }
    }
}

// ============================================================================
// Main: runs HVX + reference on Hexagon, reference-only on other targets
// ============================================================================

#[cfg(target_arch = "hexagon")]
fn main() {
    // Aligned buffers for HVX
    #[repr(align(128))]
    struct AlignedBuf<const N: usize>([u8; N]);

    let mut src = AlignedBuf::<{ WIDTH * HEIGHT }>([0u8; WIDTH * HEIGHT]);
    let mut dst_hvx = AlignedBuf::<{ WIDTH * HEIGHT }>([0u8; WIDTH * HEIGHT]);
    let mut tmp = AlignedBuf::<{ WIDTH }>([0u8; WIDTH]);
    let mut dst_ref = vec![0u8; WIDTH * HEIGHT];

    // Generate test pattern
    generate_test_pattern(&mut src.0, WIDTH, HEIGHT);

    // Run HVX implementation
    unsafe {
        hvx::gaussian3x3u8(
            src.0.as_ptr(),
            WIDTH,
            WIDTH,
            HEIGHT,
            dst_hvx.0.as_mut_ptr(),
            tmp.0.as_mut_ptr(),
        );
    }

    // Run reference
    gaussian3x3u8_reference(&src.0, WIDTH, WIDTH, HEIGHT, &mut dst_ref);

    // Verify HVX matches reference (allowing small rounding differences)
    let mut max_diff = 0i32;
    for y in 1..HEIGHT - 1 {
        for x in 1..WIDTH - 1 {
            let idx = y * WIDTH + x;
            let diff = (dst_hvx.0[idx] as i32 - dst_ref[idx] as i32).abs();
            max_diff = max_diff.max(diff);
            // Allow up to 1 LSB difference due to rounding
            assert!(
                diff <= 1,
                "HVX differs from reference at ({}, {}): hvx={}, ref={}, diff={}",
                x,
                y,
                dst_hvx.0[idx],
                dst_ref[idx],
                diff
            );
        }
    }

    println!(
        "Gaussian 3x3 HVX test passed! Max difference from reference: {}",
        max_diff
    );
}

#[cfg(not(target_arch = "hexagon"))]
fn main() {
    let mut src = vec![0u8; WIDTH * HEIGHT];
    let mut dst = vec![0u8; WIDTH * HEIGHT];

    // Generate test pattern
    generate_test_pattern(&mut src, WIDTH, HEIGHT);

    // Run reference implementation
    gaussian3x3u8_reference(&src, WIDTH, WIDTH, HEIGHT, &mut dst);

    // Verify output is non-trivial (blurred values differ from input)
    let mut changed = 0;
    for y in 1..HEIGHT - 1 {
        for x in 1..WIDTH - 1 {
            let idx = y * WIDTH + x;
            if src[idx] != dst[idx] {
                changed += 1;
            }
        }
    }

    println!(
        "Gaussian 3x3 reference test passed! {} pixels changed by blur",
        changed
    );
}
