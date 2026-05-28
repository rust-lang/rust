//! Conversions to Hexagon HVX SIMD types.

use crate::simd::*;

// HVX 128-byte mode (1024-bit vectors)
// Enable with: -C target-feature=+hvx-length128b
#[cfg(target_feature = "hvx-length128b")]
mod hvx_128b {
    use super::*;
    use core::arch::hexagon::v128::HvxVector;

    // Full vectors (1024-bit) map to HvxVector
    from_transmute! { unsafe u16x64 => HvxVector }
    from_transmute! { unsafe i16x64 => HvxVector }
    from_transmute! { unsafe u32x32 => HvxVector }
    from_transmute! { unsafe i32x32 => HvxVector }
    from_transmute! { unsafe u64x16 => HvxVector }
    from_transmute! { unsafe i64x16 => HvxVector }

    // FIXME: u8x128/i8x128 don't exist in portable-simd (max lane count is 64)
    // u8x64/i8x64 are only 512-bit (half of HvxVector in 128B mode)
}

// HVX 64-byte mode (512-bit vectors)
// Default when hvx-length128b is not specified
#[cfg(not(target_feature = "hvx-length128b"))]
mod hvx_64b {
    use super::*;
    use core::arch::hexagon::v64::HvxVector;

    // Full vectors (512-bit) map to HvxVector
    from_transmute! { unsafe u8x64 => HvxVector }
    from_transmute! { unsafe i8x64 => HvxVector }
    from_transmute! { unsafe u16x32 => HvxVector }
    from_transmute! { unsafe i16x32 => HvxVector }
    from_transmute! { unsafe u32x16 => HvxVector }
    from_transmute! { unsafe i32x16 => HvxVector }
    from_transmute! { unsafe u64x8 => HvxVector }
    from_transmute! { unsafe i64x8 => HvxVector }
}
