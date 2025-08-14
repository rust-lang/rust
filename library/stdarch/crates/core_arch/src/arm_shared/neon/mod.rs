//! ARMv7 NEON intrinsics

#[rustfmt::skip]
mod generated;
#[rustfmt::skip]
#[cfg_attr(not(target_arch = "arm"), stable(feature = "neon_intrinsics", since = "1.59.0"))]
#[cfg_attr(target_arch = "arm", unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800"))]
pub use self::generated::*;

use crate::{core_arch::simd::*, hint::unreachable_unchecked, intrinsics::simd::*, mem::transmute};
#[cfg(test)]
use stdarch_test::assert_instr;

pub(crate) trait AsUnsigned {
    type Unsigned;
    fn as_unsigned(self) -> Self::Unsigned;
}

pub(crate) trait AsSigned {
    type Signed;
    fn as_signed(self) -> Self::Signed;
}

macro_rules! impl_sign_conversions_neon {
    ($(($signed:ty, $unsigned:ty))*) => ($(
        impl AsUnsigned for $signed {
            type Unsigned = $unsigned;

            #[inline(always)]
            fn as_unsigned(self) -> $unsigned {
                unsafe { transmute(self) }
            }
        }

        impl AsSigned for $unsigned {
            type Signed = $signed;

            #[inline(always)]
            fn as_signed(self) -> $signed {
                unsafe { transmute(self) }
            }
        }
    )*)
}

pub(crate) type p8 = u8;
pub(crate) type p16 = u16;
pub(crate) type p64 = u64;
pub(crate) type p128 = u128;

types! {
    #![cfg_attr(not(target_arch = "arm"), stable(feature = "neon_intrinsics", since = "1.59.0"))]
    #![cfg_attr(target_arch = "arm", unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800"))]

    /// Arm-specific 64-bit wide vector of eight packed `i8`.
    pub struct int8x8_t(8 x pub(crate) i8);
    /// Arm-specific 64-bit wide vector of eight packed `u8`.
    pub struct uint8x8_t(8 x pub(crate) u8);
    /// Arm-specific 64-bit wide polynomial vector of eight packed `p8`.
    pub struct poly8x8_t(8 x pub(crate) p8);
    /// Arm-specific 64-bit wide vector of four packed `i16`.
    pub struct int16x4_t(4 x pub(crate) i16);
    /// Arm-specific 64-bit wide vector of four packed `u16`.
    pub struct uint16x4_t(4 x pub(crate) u16);
    /// Arm-specific 64-bit wide vector of four packed `p16`.
    pub struct poly16x4_t(4 x pub(crate) p16);
    /// Arm-specific 64-bit wide vector of two packed `i32`.
    pub struct int32x2_t(2 x pub(crate) i32);
    /// Arm-specific 64-bit wide vector of two packed `u32`.
    pub struct uint32x2_t(2 x pub(crate) u32);
    /// Arm-specific 64-bit wide vector of two packed `f32`.
    pub struct float32x2_t(2 x pub(crate) f32);
    /// Arm-specific 64-bit wide vector of one packed `i64`.
    pub struct int64x1_t(1 x pub(crate) i64);
    /// Arm-specific 64-bit wide vector of one packed `u64`.
    pub struct uint64x1_t(1 x pub(crate) u64);
    /// Arm-specific 64-bit wide vector of one packed `p64`.
    pub struct poly64x1_t(1 x pub(crate) p64);

    /// Arm-specific 128-bit wide vector of sixteen packed `i8`.
    pub struct int8x16_t(16 x pub(crate) i8);
    /// Arm-specific 128-bit wide vector of sixteen packed `u8`.
    pub struct uint8x16_t(16 x pub(crate) u8);
    /// Arm-specific 128-bit wide vector of sixteen packed `p8`.
    pub struct poly8x16_t(16 x pub(crate) p8);
    /// Arm-specific 128-bit wide vector of eight packed `i16`.
    pub struct int16x8_t(8 x pub(crate) i16);
    /// Arm-specific 128-bit wide vector of eight packed `u16`.
    pub struct uint16x8_t(8 x pub(crate) u16);
    /// Arm-specific 128-bit wide vector of eight packed `p16`.
    pub struct poly16x8_t(8 x pub(crate) p16);
    /// Arm-specific 128-bit wide vector of four packed `i32`.
    pub struct int32x4_t(4 x pub(crate) i32);
    /// Arm-specific 128-bit wide vector of four packed `u32`.
    pub struct uint32x4_t(4 x pub(crate) u32);
    /// Arm-specific 128-bit wide vector of four packed `f32`.
    pub struct float32x4_t(4 x pub(crate) f32);
    /// Arm-specific 128-bit wide vector of two packed `i64`.
    pub struct int64x2_t(2 x pub(crate) i64);
    /// Arm-specific 128-bit wide vector of two packed `u64`.
    pub struct uint64x2_t(2 x pub(crate) u64);
    /// Arm-specific 128-bit wide vector of two packed `p64`.
    pub struct poly64x2_t(2 x pub(crate) p64);
}

types! {
    #![unstable(feature = "stdarch_neon_f16", issue = "136306")]

    /// Arm-specific 64-bit wide vector of four packed `f16`.
    pub struct float16x4_t(4 x pub(crate) f16);
    /// Arm-specific 128-bit wide vector of eight packed `f16`.
    pub struct float16x8_t(8 x pub(crate) f16);
}

/// Arm-specific type containing two `int8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int8x8x2_t(pub int8x8_t, pub int8x8_t);
/// Arm-specific type containing three `int8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int8x8x3_t(pub int8x8_t, pub int8x8_t, pub int8x8_t);
/// Arm-specific type containing four `int8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int8x8x4_t(pub int8x8_t, pub int8x8_t, pub int8x8_t, pub int8x8_t);

/// Arm-specific type containing two `int8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int8x16x2_t(pub int8x16_t, pub int8x16_t);
/// Arm-specific type containing three `int8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int8x16x3_t(pub int8x16_t, pub int8x16_t, pub int8x16_t);
/// Arm-specific type containing four `int8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int8x16x4_t(pub int8x16_t, pub int8x16_t, pub int8x16_t, pub int8x16_t);

/// Arm-specific type containing two `uint8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint8x8x2_t(pub uint8x8_t, pub uint8x8_t);
/// Arm-specific type containing three `uint8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint8x8x3_t(pub uint8x8_t, pub uint8x8_t, pub uint8x8_t);
/// Arm-specific type containing four `uint8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint8x8x4_t(pub uint8x8_t, pub uint8x8_t, pub uint8x8_t, pub uint8x8_t);

/// Arm-specific type containing two `uint8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint8x16x2_t(pub uint8x16_t, pub uint8x16_t);
/// Arm-specific type containing three `uint8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint8x16x3_t(pub uint8x16_t, pub uint8x16_t, pub uint8x16_t);
/// Arm-specific type containing four `uint8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint8x16x4_t(
    pub uint8x16_t,
    pub uint8x16_t,
    pub uint8x16_t,
    pub uint8x16_t,
);

/// Arm-specific type containing two `poly8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly8x8x2_t(pub poly8x8_t, pub poly8x8_t);
/// Arm-specific type containing three `poly8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly8x8x3_t(pub poly8x8_t, pub poly8x8_t, pub poly8x8_t);
/// Arm-specific type containing four `poly8x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly8x8x4_t(pub poly8x8_t, pub poly8x8_t, pub poly8x8_t, pub poly8x8_t);

/// Arm-specific type containing two `poly8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly8x16x2_t(pub poly8x16_t, pub poly8x16_t);
/// Arm-specific type containing three `poly8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly8x16x3_t(pub poly8x16_t, pub poly8x16_t, pub poly8x16_t);
/// Arm-specific type containing four `poly8x16_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly8x16x4_t(
    pub poly8x16_t,
    pub poly8x16_t,
    pub poly8x16_t,
    pub poly8x16_t,
);

/// Arm-specific type containing two `int16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int16x4x2_t(pub int16x4_t, pub int16x4_t);
/// Arm-specific type containing three `int16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int16x4x3_t(pub int16x4_t, pub int16x4_t, pub int16x4_t);
/// Arm-specific type containing four `int16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int16x4x4_t(pub int16x4_t, pub int16x4_t, pub int16x4_t, pub int16x4_t);

/// Arm-specific type containing two `int16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int16x8x2_t(pub int16x8_t, pub int16x8_t);
/// Arm-specific type containing three `int16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int16x8x3_t(pub int16x8_t, pub int16x8_t, pub int16x8_t);
/// Arm-specific type containing four `int16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int16x8x4_t(pub int16x8_t, pub int16x8_t, pub int16x8_t, pub int16x8_t);

/// Arm-specific type containing two `uint16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint16x4x2_t(pub uint16x4_t, pub uint16x4_t);
/// Arm-specific type containing three `uint16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint16x4x3_t(pub uint16x4_t, pub uint16x4_t, pub uint16x4_t);
/// Arm-specific type containing four `uint16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint16x4x4_t(
    pub uint16x4_t,
    pub uint16x4_t,
    pub uint16x4_t,
    pub uint16x4_t,
);

/// Arm-specific type containing two `uint16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint16x8x2_t(pub uint16x8_t, pub uint16x8_t);
/// Arm-specific type containing three `uint16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint16x8x3_t(pub uint16x8_t, pub uint16x8_t, pub uint16x8_t);
/// Arm-specific type containing four `uint16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint16x8x4_t(
    pub uint16x8_t,
    pub uint16x8_t,
    pub uint16x8_t,
    pub uint16x8_t,
);

/// Arm-specific type containing two `poly16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly16x4x2_t(pub poly16x4_t, pub poly16x4_t);
/// Arm-specific type containing three `poly16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly16x4x3_t(pub poly16x4_t, pub poly16x4_t, pub poly16x4_t);
/// Arm-specific type containing four `poly16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly16x4x4_t(
    pub poly16x4_t,
    pub poly16x4_t,
    pub poly16x4_t,
    pub poly16x4_t,
);

/// Arm-specific type containing two `poly16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly16x8x2_t(pub poly16x8_t, pub poly16x8_t);
/// Arm-specific type containing three `poly16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly16x8x3_t(pub poly16x8_t, pub poly16x8_t, pub poly16x8_t);
/// Arm-specific type containing four `poly16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly16x8x4_t(
    pub poly16x8_t,
    pub poly16x8_t,
    pub poly16x8_t,
    pub poly16x8_t,
);

/// Arm-specific type containing two `int32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int32x2x2_t(pub int32x2_t, pub int32x2_t);
/// Arm-specific type containing three `int32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int32x2x3_t(pub int32x2_t, pub int32x2_t, pub int32x2_t);
/// Arm-specific type containing four `int32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int32x2x4_t(pub int32x2_t, pub int32x2_t, pub int32x2_t, pub int32x2_t);

/// Arm-specific type containing two `int32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int32x4x2_t(pub int32x4_t, pub int32x4_t);
/// Arm-specific type containing three `int32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int32x4x3_t(pub int32x4_t, pub int32x4_t, pub int32x4_t);
/// Arm-specific type containing four `int32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int32x4x4_t(pub int32x4_t, pub int32x4_t, pub int32x4_t, pub int32x4_t);

/// Arm-specific type containing two `uint32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint32x2x2_t(pub uint32x2_t, pub uint32x2_t);
/// Arm-specific type containing three `uint32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint32x2x3_t(pub uint32x2_t, pub uint32x2_t, pub uint32x2_t);
/// Arm-specific type containing four `uint32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint32x2x4_t(
    pub uint32x2_t,
    pub uint32x2_t,
    pub uint32x2_t,
    pub uint32x2_t,
);

/// Arm-specific type containing two `uint32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint32x4x2_t(pub uint32x4_t, pub uint32x4_t);
/// Arm-specific type containing three `uint32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint32x4x3_t(pub uint32x4_t, pub uint32x4_t, pub uint32x4_t);
/// Arm-specific type containing four `uint32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint32x4x4_t(
    pub uint32x4_t,
    pub uint32x4_t,
    pub uint32x4_t,
    pub uint32x4_t,
);

/// Arm-specific type containing two `float16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "stdarch_neon_f16", issue = "136306")]
pub struct float16x4x2_t(pub float16x4_t, pub float16x4_t);

/// Arm-specific type containing three `float16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "stdarch_neon_f16", issue = "136306")]
pub struct float16x4x3_t(pub float16x4_t, pub float16x4_t, pub float16x4_t);

/// Arm-specific type containing four `float16x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "stdarch_neon_f16", issue = "136306")]
pub struct float16x4x4_t(
    pub float16x4_t,
    pub float16x4_t,
    pub float16x4_t,
    pub float16x4_t,
);

/// Arm-specific type containing two `float16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "stdarch_neon_f16", issue = "136306")]
pub struct float16x8x2_t(pub float16x8_t, pub float16x8_t);

/// Arm-specific type containing three `float16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "stdarch_neon_f16", issue = "136306")]
pub struct float16x8x3_t(pub float16x8_t, pub float16x8_t, pub float16x8_t);

/// Arm-specific type containing four `float16x8_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[unstable(feature = "stdarch_neon_f16", issue = "136306")]
pub struct float16x8x4_t(
    pub float16x8_t,
    pub float16x8_t,
    pub float16x8_t,
    pub float16x8_t,
);

/// Arm-specific type containing two `float32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct float32x2x2_t(pub float32x2_t, pub float32x2_t);
/// Arm-specific type containing three `float32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct float32x2x3_t(pub float32x2_t, pub float32x2_t, pub float32x2_t);
/// Arm-specific type containing four `float32x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct float32x2x4_t(
    pub float32x2_t,
    pub float32x2_t,
    pub float32x2_t,
    pub float32x2_t,
);

/// Arm-specific type containing two `float32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct float32x4x2_t(pub float32x4_t, pub float32x4_t);
/// Arm-specific type containing three `float32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct float32x4x3_t(pub float32x4_t, pub float32x4_t, pub float32x4_t);
/// Arm-specific type containing four `float32x4_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct float32x4x4_t(
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
);

/// Arm-specific type containing two `int64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int64x1x2_t(pub int64x1_t, pub int64x1_t);
/// Arm-specific type containing three `int64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int64x1x3_t(pub int64x1_t, pub int64x1_t, pub int64x1_t);
/// Arm-specific type containing four `int64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int64x1x4_t(pub int64x1_t, pub int64x1_t, pub int64x1_t, pub int64x1_t);

/// Arm-specific type containing two `int64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int64x2x2_t(pub int64x2_t, pub int64x2_t);
/// Arm-specific type containing three `int64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int64x2x3_t(pub int64x2_t, pub int64x2_t, pub int64x2_t);
/// Arm-specific type containing four `int64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct int64x2x4_t(pub int64x2_t, pub int64x2_t, pub int64x2_t, pub int64x2_t);

/// Arm-specific type containing two `uint64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint64x1x2_t(pub uint64x1_t, pub uint64x1_t);
/// Arm-specific type containing three `uint64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint64x1x3_t(pub uint64x1_t, pub uint64x1_t, pub uint64x1_t);
/// Arm-specific type containing four `uint64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint64x1x4_t(
    pub uint64x1_t,
    pub uint64x1_t,
    pub uint64x1_t,
    pub uint64x1_t,
);

/// Arm-specific type containing two `uint64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint64x2x2_t(pub uint64x2_t, pub uint64x2_t);
/// Arm-specific type containing three `uint64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint64x2x3_t(pub uint64x2_t, pub uint64x2_t, pub uint64x2_t);
/// Arm-specific type containing four `uint64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct uint64x2x4_t(
    pub uint64x2_t,
    pub uint64x2_t,
    pub uint64x2_t,
    pub uint64x2_t,
);

/// Arm-specific type containing two `poly64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly64x1x2_t(pub poly64x1_t, pub poly64x1_t);
/// Arm-specific type containing three `poly64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly64x1x3_t(pub poly64x1_t, pub poly64x1_t, pub poly64x1_t);
/// Arm-specific type containing four `poly64x1_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly64x1x4_t(
    pub poly64x1_t,
    pub poly64x1_t,
    pub poly64x1_t,
    pub poly64x1_t,
);

/// Arm-specific type containing two `poly64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly64x2x2_t(pub poly64x2_t, pub poly64x2_t);
/// Arm-specific type containing three `poly64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly64x2x3_t(pub poly64x2_t, pub poly64x2_t, pub poly64x2_t);
/// Arm-specific type containing four `poly64x2_t` vectors.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    not(target_arch = "arm"),
    stable(feature = "neon_intrinsics", since = "1.59.0")
)]
#[cfg_attr(
    target_arch = "arm",
    unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")
)]
pub struct poly64x2x4_t(
    pub poly64x2_t,
    pub poly64x2_t,
    pub poly64x2_t,
    pub poly64x2_t,
);

impl_sign_conversions_neon! {
    (i8, u8)
    (i16, u16)
    (i32, u32)
    (i64, u64)
    (*const i8, *const u8)
    (*const i16, *const u16)
    (*const i32, *const u32)
    (*const i64, *const u64)
    (*mut i8, *mut u8)
    (*mut i16, *mut u16)
    (*mut i32, *mut u32)
    (*mut i64, *mut u64)
    (int16x4_t, uint16x4_t)
    (int16x8_t, uint16x8_t)
    (int32x2_t, uint32x2_t)
    (int32x4_t, uint32x4_t)
    (int64x1_t, uint64x1_t)
    (int64x2_t, uint64x2_t)
    (int8x16_t, uint8x16_t)
    (int8x8_t, uint8x8_t)
    (uint16x4_t, int16x4_t)
    (uint16x8_t, int16x8_t)
    (uint32x2_t, int32x2_t)
    (uint32x4_t, int32x4_t)
    (uint64x1_t, int64x1_t)
    (uint64x2_t, int64x2_t)
    (uint8x16_t, int8x16_t)
    (uint8x8_t, int8x8_t)
    (int16x4x2_t, uint16x4x2_t)
    (int16x4x3_t, uint16x4x3_t)
    (int16x4x4_t, uint16x4x4_t)
    (int16x8x2_t, uint16x8x2_t)
    (int16x8x3_t, uint16x8x3_t)
    (int16x8x4_t, uint16x8x4_t)
    (int32x2x2_t, uint32x2x2_t)
    (int32x2x3_t, uint32x2x3_t)
    (int32x2x4_t, uint32x2x4_t)
    (int32x4x2_t, uint32x4x2_t)
    (int32x4x3_t, uint32x4x3_t)
    (int32x4x4_t, uint32x4x4_t)
    (int64x1x2_t, uint64x1x2_t)
    (int64x1x3_t, uint64x1x3_t)
    (int64x1x4_t, uint64x1x4_t)
    (int64x2x2_t, uint64x2x2_t)
    (int64x2x3_t, uint64x2x3_t)
    (int64x2x4_t, uint64x2x4_t)
    (int8x16x2_t, uint8x16x2_t)
    (int8x16x3_t, uint8x16x3_t)
    (int8x16x4_t, uint8x16x4_t)
    (int8x8x2_t, uint8x8x2_t)
    (int8x8x3_t, uint8x8x3_t)
    (int8x8x4_t, uint8x8x4_t)
    (uint16x4x2_t, int16x4x2_t)
    (uint16x4x3_t, int16x4x3_t)
    (uint16x4x4_t, int16x4x4_t)
    (uint16x8x2_t, int16x8x2_t)
    (uint16x8x3_t, int16x8x3_t)
    (uint16x8x4_t, int16x8x4_t)
    (uint32x2x2_t, int32x2x2_t)
    (uint32x2x3_t, int32x2x3_t)
    (uint32x2x4_t, int32x2x4_t)
    (uint32x4x2_t, int32x4x2_t)
    (uint32x4x3_t, int32x4x3_t)
    (uint32x4x4_t, int32x4x4_t)
    (uint64x1x2_t, int64x1x2_t)
    (uint64x1x3_t, int64x1x3_t)
    (uint64x1x4_t, int64x1x4_t)
    (uint64x2x2_t, int64x2x2_t)
    (uint64x2x3_t, int64x2x3_t)
    (uint64x2x4_t, int64x2x4_t)
    (uint8x16x2_t, int8x16x2_t)
    (uint8x16x3_t, int8x16x3_t)
    (uint8x16x4_t, int8x16x4_t)
    (uint8x8x2_t, int8x8x2_t)
    (uint8x8x3_t, int8x8x3_t)
    (uint8x8x4_t, int8x8x4_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    use crate::core_arch::aarch64::*;
    #[cfg(target_arch = "arm")]
    use crate::core_arch::arm::*;
    use crate::core_arch::arm_shared::test_support::*;
    use crate::core_arch::simd::*;
    use std::{mem::transmute, vec::Vec};
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let elem: i8 = 42;
        let e = i8x8::new(0, 1, 2, 3, 4, 5, 6, 42);
        let r: i8x8 = transmute(vld1_lane_s8::<7>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let elem: i8 = 42;
        let e = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 42);
        let r: i8x16 = transmute(vld1q_lane_s8::<15>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let elem: i16 = 42;
        let e = i16x4::new(0, 1, 2, 42);
        let r: i16x4 = transmute(vld1_lane_s16::<3>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let elem: i16 = 42;
        let e = i16x8::new(0, 1, 2, 3, 4, 5, 6, 42);
        let r: i16x8 = transmute(vld1q_lane_s16::<7>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_s32() {
        let a = i32x2::new(0, 1);
        let elem: i32 = 42;
        let e = i32x2::new(0, 42);
        let r: i32x2 = transmute(vld1_lane_s32::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let elem: i32 = 42;
        let e = i32x4::new(0, 1, 2, 42);
        let r: i32x4 = transmute(vld1q_lane_s32::<3>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_s64() {
        let a = i64x1::new(0);
        let elem: i64 = 42;
        let e = i64x1::new(42);
        let r: i64x1 = transmute(vld1_lane_s64::<0>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_s64() {
        let a = i64x2::new(0, 1);
        let elem: i64 = 42;
        let e = i64x2::new(0, 42);
        let r: i64x2 = transmute(vld1q_lane_s64::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let elem: u8 = 42;
        let e = u8x8::new(0, 1, 2, 3, 4, 5, 6, 42);
        let r: u8x8 = transmute(vld1_lane_u8::<7>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let elem: u8 = 42;
        let e = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 42);
        let r: u8x16 = transmute(vld1q_lane_u8::<15>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let elem: u16 = 42;
        let e = u16x4::new(0, 1, 2, 42);
        let r: u16x4 = transmute(vld1_lane_u16::<3>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let elem: u16 = 42;
        let e = u16x8::new(0, 1, 2, 3, 4, 5, 6, 42);
        let r: u16x8 = transmute(vld1q_lane_u16::<7>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_u32() {
        let a = u32x2::new(0, 1);
        let elem: u32 = 42;
        let e = u32x2::new(0, 42);
        let r: u32x2 = transmute(vld1_lane_u32::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let elem: u32 = 42;
        let e = u32x4::new(0, 1, 2, 42);
        let r: u32x4 = transmute(vld1q_lane_u32::<3>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_u64() {
        let a = u64x1::new(0);
        let elem: u64 = 42;
        let e = u64x1::new(42);
        let r: u64x1 = transmute(vld1_lane_u64::<0>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_u64() {
        let a = u64x2::new(0, 1);
        let elem: u64 = 42;
        let e = u64x2::new(0, 42);
        let r: u64x2 = transmute(vld1q_lane_u64::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_p8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let elem: p8 = 42;
        let e = u8x8::new(0, 1, 2, 3, 4, 5, 6, 42);
        let r: u8x8 = transmute(vld1_lane_p8::<7>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_p8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let elem: p8 = 42;
        let e = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 42);
        let r: u8x16 = transmute(vld1q_lane_p8::<15>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_p16() {
        let a = u16x4::new(0, 1, 2, 3);
        let elem: p16 = 42;
        let e = u16x4::new(0, 1, 2, 42);
        let r: u16x4 = transmute(vld1_lane_p16::<3>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_p16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let elem: p16 = 42;
        let e = u16x8::new(0, 1, 2, 3, 4, 5, 6, 42);
        let r: u16x8 = transmute(vld1q_lane_p16::<7>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon,aes")]
    unsafe fn test_vld1_lane_p64() {
        let a = u64x1::new(0);
        let elem: u64 = 42;
        let e = u64x1::new(42);
        let r: u64x1 = transmute(vld1_lane_p64::<0>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon,aes")]
    unsafe fn test_vld1q_lane_p64() {
        let a = u64x2::new(0, 1);
        let elem: u64 = 42;
        let e = u64x2::new(0, 42);
        let r: u64x2 = transmute(vld1q_lane_p64::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_lane_f32() {
        let a = f32x2::new(0., 1.);
        let elem: f32 = 42.;
        let e = f32x2::new(0., 42.);
        let r: f32x2 = transmute(vld1_lane_f32::<1>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_lane_f32() {
        let a = f32x4::new(0., 1., 2., 3.);
        let elem: f32 = 42.;
        let e = f32x4::new(0., 1., 2., 42.);
        let r: f32x4 = transmute(vld1q_lane_f32::<3>(&elem, transmute(a)));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_s8() {
        let elem: i8 = 42;
        let e = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let r: i8x8 = transmute(vld1_dup_s8(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_s8() {
        let elem: i8 = 42;
        let e = i8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: i8x16 = transmute(vld1q_dup_s8(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_s16() {
        let elem: i16 = 42;
        let e = i16x4::new(42, 42, 42, 42);
        let r: i16x4 = transmute(vld1_dup_s16(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_s16() {
        let elem: i16 = 42;
        let e = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let r: i16x8 = transmute(vld1q_dup_s16(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_s32() {
        let elem: i32 = 42;
        let e = i32x2::new(42, 42);
        let r: i32x2 = transmute(vld1_dup_s32(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_s32() {
        let elem: i32 = 42;
        let e = i32x4::new(42, 42, 42, 42);
        let r: i32x4 = transmute(vld1q_dup_s32(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_s64() {
        let elem: i64 = 42;
        let e = i64x1::new(42);
        let r: i64x1 = transmute(vld1_dup_s64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_s64() {
        let elem: i64 = 42;
        let e = i64x2::new(42, 42);
        let r: i64x2 = transmute(vld1q_dup_s64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_u8() {
        let elem: u8 = 42;
        let e = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let r: u8x8 = transmute(vld1_dup_u8(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_u8() {
        let elem: u8 = 42;
        let e = u8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: u8x16 = transmute(vld1q_dup_u8(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_u16() {
        let elem: u16 = 42;
        let e = u16x4::new(42, 42, 42, 42);
        let r: u16x4 = transmute(vld1_dup_u16(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_u16() {
        let elem: u16 = 42;
        let e = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let r: u16x8 = transmute(vld1q_dup_u16(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_u32() {
        let elem: u32 = 42;
        let e = u32x2::new(42, 42);
        let r: u32x2 = transmute(vld1_dup_u32(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_u32() {
        let elem: u32 = 42;
        let e = u32x4::new(42, 42, 42, 42);
        let r: u32x4 = transmute(vld1q_dup_u32(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_u64() {
        let elem: u64 = 42;
        let e = u64x1::new(42);
        let r: u64x1 = transmute(vld1_dup_u64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_u64() {
        let elem: u64 = 42;
        let e = u64x2::new(42, 42);
        let r: u64x2 = transmute(vld1q_dup_u64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_p8() {
        let elem: p8 = 42;
        let e = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let r: u8x8 = transmute(vld1_dup_p8(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_p8() {
        let elem: p8 = 42;
        let e = u8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: u8x16 = transmute(vld1q_dup_p8(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_p16() {
        let elem: p16 = 42;
        let e = u16x4::new(42, 42, 42, 42);
        let r: u16x4 = transmute(vld1_dup_p16(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_p16() {
        let elem: p16 = 42;
        let e = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let r: u16x8 = transmute(vld1q_dup_p16(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon,aes")]
    unsafe fn test_vld1_dup_p64() {
        let elem: u64 = 42;
        let e = u64x1::new(42);
        let r: u64x1 = transmute(vld1_dup_p64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon,aes")]
    unsafe fn test_vld1q_dup_p64() {
        let elem: u64 = 42;
        let e = u64x2::new(42, 42);
        let r: u64x2 = transmute(vld1q_dup_p64(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1_dup_f32() {
        let elem: f32 = 42.;
        let e = f32x2::new(42., 42.);
        let r: f32x2 = transmute(vld1_dup_f32(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vld1q_dup_f32() {
        let elem: f32 = 42.;
        let e = f32x4::new(42., 42., 42., 42.);
        let r: f32x4 = transmute(vld1q_dup_f32(&elem));
        assert_eq!(r, e)
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u8() {
        let v = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = vget_lane_u8::<1>(transmute(v));
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u32() {
        let v = i32x4::new(1, 2, 3, 4);
        let r = vgetq_lane_u32::<1>(transmute(v));
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_s32() {
        let v = i32x4::new(1, 2, 3, 4);
        let r = vgetq_lane_s32::<1>(transmute(v));
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u64() {
        let v: u64 = 1;
        let r = vget_lane_u64::<0>(transmute(v));
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u16() {
        let v = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r = vgetq_lane_u16::<1>(transmute(v));
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_s8() {
        let v = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = vget_lane_s8::<2>(transmute(v));
        assert_eq!(r, 2);
        let r = vget_lane_s8::<4>(transmute(v));
        assert_eq!(r, 4);
        let r = vget_lane_s8::<5>(transmute(v));
        assert_eq!(r, 5);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_p8() {
        let v = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = vget_lane_p8::<2>(transmute(v));
        assert_eq!(r, 2);
        let r = vget_lane_p8::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vget_lane_p8::<5>(transmute(v));
        assert_eq!(r, 5);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_p16() {
        let v = u16x4::new(0, 1, 2, 3);
        let r = vget_lane_p16::<2>(transmute(v));
        assert_eq!(r, 2);
        let r = vget_lane_p16::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vget_lane_p16::<0>(transmute(v));
        assert_eq!(r, 0);
        let r = vget_lane_p16::<1>(transmute(v));
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_s16() {
        let v = i16x4::new(0, 1, 2, 3);
        let r = vget_lane_s16::<2>(transmute(v));
        assert_eq!(r, 2);
        let r = vget_lane_s16::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vget_lane_s16::<0>(transmute(v));
        assert_eq!(r, 0);
        let r = vget_lane_s16::<1>(transmute(v));
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u16() {
        let v = u16x4::new(0, 1, 2, 3);
        let r = vget_lane_u16::<2>(transmute(v));
        assert_eq!(r, 2);
        let r = vget_lane_u16::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vget_lane_u16::<0>(transmute(v));
        assert_eq!(r, 0);
        let r = vget_lane_u16::<1>(transmute(v));
        assert_eq!(r, 1);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_f32() {
        let v = f32x2::new(0.0, 1.0);
        let r = vget_lane_f32::<1>(transmute(v));
        assert_eq!(r, 1.0);
        let r = vget_lane_f32::<0>(transmute(v));
        assert_eq!(r, 0.0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_s32() {
        let v = i32x2::new(0, 1);
        let r = vget_lane_s32::<1>(transmute(v));
        assert_eq!(r, 1);
        let r = vget_lane_s32::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_u32() {
        let v = u32x2::new(0, 1);
        let r = vget_lane_u32::<1>(transmute(v));
        assert_eq!(r, 1);
        let r = vget_lane_u32::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_s64() {
        let v = i64x1::new(1);
        let r = vget_lane_s64::<0>(transmute(v));
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_lane_p64() {
        let v = u64x1::new(1);
        let r = vget_lane_p64::<0>(transmute(v));
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_s8() {
        let v = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = vgetq_lane_s8::<7>(transmute(v));
        assert_eq!(r, 7);
        let r = vgetq_lane_s8::<13>(transmute(v));
        assert_eq!(r, 13);
        let r = vgetq_lane_s8::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vgetq_lane_s8::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_p8() {
        let v = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = vgetq_lane_p8::<7>(transmute(v));
        assert_eq!(r, 7);
        let r = vgetq_lane_p8::<13>(transmute(v));
        assert_eq!(r, 13);
        let r = vgetq_lane_p8::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vgetq_lane_p8::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u8() {
        let v = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = vgetq_lane_u8::<7>(transmute(v));
        assert_eq!(r, 7);
        let r = vgetq_lane_u8::<13>(transmute(v));
        assert_eq!(r, 13);
        let r = vgetq_lane_u8::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vgetq_lane_u8::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_s16() {
        let v = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = vgetq_lane_s16::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vgetq_lane_s16::<6>(transmute(v));
        assert_eq!(r, 6);
        let r = vgetq_lane_s16::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_p16() {
        let v = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = vgetq_lane_p16::<3>(transmute(v));
        assert_eq!(r, 3);
        let r = vgetq_lane_p16::<7>(transmute(v));
        assert_eq!(r, 7);
        let r = vgetq_lane_p16::<1>(transmute(v));
        assert_eq!(r, 1);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_f32() {
        let v = f32x4::new(0.0, 1.0, 2.0, 3.0);
        let r = vgetq_lane_f32::<3>(transmute(v));
        assert_eq!(r, 3.0);
        let r = vgetq_lane_f32::<0>(transmute(v));
        assert_eq!(r, 0.0);
        let r = vgetq_lane_f32::<2>(transmute(v));
        assert_eq!(r, 2.0);
        let r = vgetq_lane_f32::<1>(transmute(v));
        assert_eq!(r, 1.0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_s64() {
        let v = i64x2::new(0, 1);
        let r = vgetq_lane_s64::<1>(transmute(v));
        assert_eq!(r, 1);
        let r = vgetq_lane_s64::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_p64() {
        let v = u64x2::new(0, 1);
        let r = vgetq_lane_p64::<1>(transmute(v));
        assert_eq!(r, 1);
        let r = vgetq_lane_p64::<0>(transmute(v));
        assert_eq!(r, 0);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_s64() {
        let a: i64x1 = i64x1::new(0);
        let b: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vext_s64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_u64() {
        let a: u64x1 = u64x1::new(0);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vext_u64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = i8x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x8 = transmute(vget_high_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = i16x4::new(5, 6, 7, 8);
        let r: i16x4 = transmute(vget_high_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let e = i32x2::new(3, 4);
        let r: i32x2 = transmute(vget_high_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_s64() {
        let a = i64x2::new(1, 2);
        let e = i64x1::new(2);
        let r: i64x1 = transmute(vget_high_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x8 = transmute(vget_high_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u16x4::new(5, 6, 7, 8);
        let r: u16x4 = transmute(vget_high_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let e = u32x2::new(3, 4);
        let r: u32x2 = transmute(vget_high_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_u64() {
        let a = u64x2::new(1, 2);
        let e = u64x1::new(2);
        let r: u64x1 = transmute(vget_high_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_p8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x8 = transmute(vget_high_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_p16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u16x4::new(5, 6, 7, 8);
        let r: u16x4 = transmute(vget_high_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_high_f32() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let e = f32x2::new(3.0, 4.0);
        let r: f32x2 = transmute(vget_high_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vget_low_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vget_low_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let e = i32x2::new(1, 2);
        let r: i32x2 = transmute(vget_low_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_s64() {
        let a = i64x2::new(1, 2);
        let e = i64x1::new(1);
        let r: i64x1 = transmute(vget_low_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vget_low_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vget_low_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let e = u32x2::new(1, 2);
        let r: u32x2 = transmute(vget_low_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_u64() {
        let a = u64x2::new(1, 2);
        let e = u64x1::new(1);
        let r: u64x1 = transmute(vget_low_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_p8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vget_low_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_p16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vget_low_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vget_low_f32() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let e = f32x2::new(1.0, 2.0);
        let r: f32x2 = transmute(vget_low_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_s8() {
        let v: i8 = 42;
        let e = i8x16::new(
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        );
        let r: i8x16 = transmute(vdupq_n_s8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_s16() {
        let v: i16 = 64;
        let e = i16x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: i16x8 = transmute(vdupq_n_s16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_s32() {
        let v: i32 = 64;
        let e = i32x4::new(64, 64, 64, 64);
        let r: i32x4 = transmute(vdupq_n_s32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_s64() {
        let v: i64 = 64;
        let e = i64x2::new(64, 64);
        let r: i64x2 = transmute(vdupq_n_s64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_u8() {
        let v: u8 = 64;
        let e = u8x16::new(
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        );
        let r: u8x16 = transmute(vdupq_n_u8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_u16() {
        let v: u16 = 64;
        let e = u16x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u16x8 = transmute(vdupq_n_u16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_u32() {
        let v: u32 = 64;
        let e = u32x4::new(64, 64, 64, 64);
        let r: u32x4 = transmute(vdupq_n_u32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_u64() {
        let v: u64 = 64;
        let e = u64x2::new(64, 64);
        let r: u64x2 = transmute(vdupq_n_u64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_p8() {
        let v: p8 = 64;
        let e = u8x16::new(
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        );
        let r: u8x16 = transmute(vdupq_n_p8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_p16() {
        let v: p16 = 64;
        let e = u16x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u16x8 = transmute(vdupq_n_p16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_n_f32() {
        let v: f32 = 64.0;
        let e = f32x4::new(64.0, 64.0, 64.0, 64.0);
        let r: f32x4 = transmute(vdupq_n_f32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_s8() {
        let v: i8 = 64;
        let e = i8x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: i8x8 = transmute(vdup_n_s8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_s16() {
        let v: i16 = 64;
        let e = i16x4::new(64, 64, 64, 64);
        let r: i16x4 = transmute(vdup_n_s16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_s32() {
        let v: i32 = 64;
        let e = i32x2::new(64, 64);
        let r: i32x2 = transmute(vdup_n_s32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_s64() {
        let v: i64 = 64;
        let e = i64x1::new(64);
        let r: i64x1 = transmute(vdup_n_s64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_u8() {
        let v: u8 = 64;
        let e = u8x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u8x8 = transmute(vdup_n_u8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_u16() {
        let v: u16 = 64;
        let e = u16x4::new(64, 64, 64, 64);
        let r: u16x4 = transmute(vdup_n_u16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_u32() {
        let v: u32 = 64;
        let e = u32x2::new(64, 64);
        let r: u32x2 = transmute(vdup_n_u32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_u64() {
        let v: u64 = 64;
        let e = u64x1::new(64);
        let r: u64x1 = transmute(vdup_n_u64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_p8() {
        let v: p8 = 64;
        let e = u8x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u8x8 = transmute(vdup_n_p8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_p16() {
        let v: p16 = 64;
        let e = u16x4::new(64, 64, 64, 64);
        let r: u16x4 = transmute(vdup_n_p16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_n_f32() {
        let v: f32 = 64.0;
        let e = f32x2::new(64.0, 64.0);
        let r: f32x2 = transmute(vdup_n_f32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vldrq_p128() {
        let v: [p128; 2] = [1, 2];
        let e: p128 = 2;
        let r: p128 = vldrq_p128(v[1..].as_ptr());
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vstrq_p128() {
        let v: [p128; 2] = [1, 2];
        let e: p128 = 2;
        let mut r: p128 = 1;
        vstrq_p128(&mut r, v[1]);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_s8() {
        let v: i8 = 64;
        let e = i8x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: i8x8 = transmute(vmov_n_s8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_s16() {
        let v: i16 = 64;
        let e = i16x4::new(64, 64, 64, 64);
        let r: i16x4 = transmute(vmov_n_s16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_s32() {
        let v: i32 = 64;
        let e = i32x2::new(64, 64);
        let r: i32x2 = transmute(vmov_n_s32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_s64() {
        let v: i64 = 64;
        let e = i64x1::new(64);
        let r: i64x1 = transmute(vmov_n_s64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_u8() {
        let v: u8 = 64;
        let e = u8x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u8x8 = transmute(vmov_n_u8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_u16() {
        let v: u16 = 64;
        let e = u16x4::new(64, 64, 64, 64);
        let r: u16x4 = transmute(vmov_n_u16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_u32() {
        let v: u32 = 64;
        let e = u32x2::new(64, 64);
        let r: u32x2 = transmute(vmov_n_u32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_u64() {
        let v: u64 = 64;
        let e = u64x1::new(64);
        let r: u64x1 = transmute(vmov_n_u64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_p8() {
        let v: p8 = 64;
        let e = u8x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u8x8 = transmute(vmov_n_p8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_p16() {
        let v: p16 = 64;
        let e = u16x4::new(64, 64, 64, 64);
        let r: u16x4 = transmute(vmov_n_p16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmov_n_f32() {
        let v: f32 = 64.0;
        let e = f32x2::new(64.0, 64.0);
        let r: f32x2 = transmute(vmov_n_f32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_s8() {
        let v: i8 = 64;
        let e = i8x16::new(
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        );
        let r: i8x16 = transmute(vmovq_n_s8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_s16() {
        let v: i16 = 64;
        let e = i16x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: i16x8 = transmute(vmovq_n_s16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_s32() {
        let v: i32 = 64;
        let e = i32x4::new(64, 64, 64, 64);
        let r: i32x4 = transmute(vmovq_n_s32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_s64() {
        let v: i64 = 64;
        let e = i64x2::new(64, 64);
        let r: i64x2 = transmute(vmovq_n_s64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_u8() {
        let v: u8 = 64;
        let e = u8x16::new(
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        );
        let r: u8x16 = transmute(vmovq_n_u8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_u16() {
        let v: u16 = 64;
        let e = u16x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u16x8 = transmute(vmovq_n_u16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_u32() {
        let v: u32 = 64;
        let e = u32x4::new(64, 64, 64, 64);
        let r: u32x4 = transmute(vmovq_n_u32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_u64() {
        let v: u64 = 64;
        let e = u64x2::new(64, 64);
        let r: u64x2 = transmute(vmovq_n_u64(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_p8() {
        let v: p8 = 64;
        let e = u8x16::new(
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        );
        let r: u8x16 = transmute(vmovq_n_p8(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_p16() {
        let v: p16 = 64;
        let e = u16x8::new(64, 64, 64, 64, 64, 64, 64, 64);
        let r: u16x8 = transmute(vmovq_n_p16(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovq_n_f32() {
        let v: f32 = 64.0;
        let e = f32x4::new(64.0, 64.0, 64.0, 64.0);
        let r: f32x4 = transmute(vmovq_n_f32(v));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vgetq_lane_u64() {
        let v = i64x2::new(1, 2);
        let r = vgetq_lane_u64::<1>(transmute(v));
        assert_eq!(r, 2);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s8() {
        test_ari_s8(
            |i, j| vadd_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s8() {
        testq_ari_s8(
            |i, j| vaddq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s16() {
        test_ari_s16(
            |i, j| vadd_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s16() {
        testq_ari_s16(
            |i, j| vaddq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_s32() {
        test_ari_s32(
            |i, j| vadd_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_s32() {
        testq_ari_s32(
            |i, j| vaddq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_add(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u8() {
        test_ari_u8(
            |i, j| vadd_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u8() {
        testq_ari_u8(
            |i, j| vaddq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u16() {
        test_ari_u16(
            |i, j| vadd_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u16() {
        testq_ari_u16(
            |i, j| vaddq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_u32() {
        test_ari_u32(
            |i, j| vadd_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_add(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_u32() {
        testq_ari_u32(
            |i, j| vaddq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_add(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vadd_f32() {
        test_ari_f32(|i, j| vadd_f32(i, j), |a: f32, b: f32| -> f32 { a + b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaddq_f32() {
        testq_ari_f32(|i, j| vaddq_f32(i, j), |a: f32, b: f32| -> f32 { a + b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_s8() {
        let v = i8::MAX;
        let a = i8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as i16);
        let e = i16x8::new(v, v, v, v, v, v, v, v);
        let r: i16x8 = transmute(vaddl_s8(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_s16() {
        let v = i16::MAX;
        let a = i16x4::new(v, v, v, v);
        let v = 2 * (v as i32);
        let e = i32x4::new(v, v, v, v);
        let r: i32x4 = transmute(vaddl_s16(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_s32() {
        let v = i32::MAX;
        let a = i32x2::new(v, v);
        let v = 2 * (v as i64);
        let e = i64x2::new(v, v);
        let r: i64x2 = transmute(vaddl_s32(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_u8() {
        let v = u8::MAX;
        let a = u8x8::new(v, v, v, v, v, v, v, v);
        let v = 2 * (v as u16);
        let e = u16x8::new(v, v, v, v, v, v, v, v);
        let r: u16x8 = transmute(vaddl_u8(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_u16() {
        let v = u16::MAX;
        let a = u16x4::new(v, v, v, v);
        let v = 2 * (v as u32);
        let e = u32x4::new(v, v, v, v);
        let r: u32x4 = transmute(vaddl_u16(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_u32() {
        let v = u32::MAX;
        let a = u32x2::new(v, v);
        let v = 2 * (v as u64);
        let e = u64x2::new(v, v);
        let r: u64x2 = transmute(vaddl_u32(transmute(a), transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_high_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let x = i8::MAX;
        let b = i8x16::new(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x);
        let x = x as i16;
        let e = i16x8::new(x + 8, x + 9, x + 10, x + 11, x + 12, x + 13, x + 14, x + 15);
        let r: i16x8 = transmute(vaddl_high_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_high_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let x = i16::MAX;
        let b = i16x8::new(x, x, x, x, x, x, x, x);
        let x = x as i32;
        let e = i32x4::new(x + 4, x + 5, x + 6, x + 7);
        let r: i32x4 = transmute(vaddl_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_high_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let x = i32::MAX;
        let b = i32x4::new(x, x, x, x);
        let x = x as i64;
        let e = i64x2::new(x + 2, x + 3);
        let r: i64x2 = transmute(vaddl_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_high_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let x = u8::MAX;
        let b = u8x16::new(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x);
        let x = x as u16;
        let e = u16x8::new(x + 8, x + 9, x + 10, x + 11, x + 12, x + 13, x + 14, x + 15);
        let r: u16x8 = transmute(vaddl_high_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_high_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let x = u16::MAX;
        let b = u16x8::new(x, x, x, x, x, x, x, x);
        let x = x as u32;
        let e = u32x4::new(x + 4, x + 5, x + 6, x + 7);
        let r: u32x4 = transmute(vaddl_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddl_high_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let x = u32::MAX;
        let b = u32x4::new(x, x, x, x);
        let x = x as u64;
        let e = u64x2::new(x + 2, x + 3);
        let r: u64x2 = transmute(vaddl_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_s8() {
        let x = i16::MAX;
        let a = i16x8::new(x, 1, 2, 3, 4, 5, 6, 7);
        let y = i8::MAX;
        let b = i8x8::new(y, y, y, y, y, y, y, y);
        let y = y as i16;
        let e = i16x8::new(
            x.wrapping_add(y),
            1 + y,
            2 + y,
            3 + y,
            4 + y,
            5 + y,
            6 + y,
            7 + y,
        );
        let r: i16x8 = transmute(vaddw_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_s16() {
        let x = i32::MAX;
        let a = i32x4::new(x, 1, 2, 3);
        let y = i16::MAX;
        let b = i16x4::new(y, y, y, y);
        let y = y as i32;
        let e = i32x4::new(x.wrapping_add(y), 1 + y, 2 + y, 3 + y);
        let r: i32x4 = transmute(vaddw_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_s32() {
        let x = i64::MAX;
        let a = i64x2::new(x, 1);
        let y = i32::MAX;
        let b = i32x2::new(y, y);
        let y = y as i64;
        let e = i64x2::new(x.wrapping_add(y), 1 + y);
        let r: i64x2 = transmute(vaddw_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_u8() {
        let x = u16::MAX;
        let a = u16x8::new(x, 1, 2, 3, 4, 5, 6, 7);
        let y = u8::MAX;
        let b = u8x8::new(y, y, y, y, y, y, y, y);
        let y = y as u16;
        let e = u16x8::new(
            x.wrapping_add(y),
            1 + y,
            2 + y,
            3 + y,
            4 + y,
            5 + y,
            6 + y,
            7 + y,
        );
        let r: u16x8 = transmute(vaddw_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_u16() {
        let x = u32::MAX;
        let a = u32x4::new(x, 1, 2, 3);
        let y = u16::MAX;
        let b = u16x4::new(y, y, y, y);
        let y = y as u32;
        let e = u32x4::new(x.wrapping_add(y), 1 + y, 2 + y, 3 + y);
        let r: u32x4 = transmute(vaddw_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_u32() {
        let x = u64::MAX;
        let a = u64x2::new(x, 1);
        let y = u32::MAX;
        let b = u32x2::new(y, y);
        let y = y as u64;
        let e = u64x2::new(x.wrapping_add(y), 1 + y);
        let r: u64x2 = transmute(vaddw_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_high_s8() {
        let x = i16::MAX;
        let a = i16x8::new(x, 1, 2, 3, 4, 5, 6, 7);
        let y = i8::MAX;
        let b = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, y, y, y, y, y, y, y, y);
        let y = y as i16;
        let e = i16x8::new(
            x.wrapping_add(y),
            1 + y,
            2 + y,
            3 + y,
            4 + y,
            5 + y,
            6 + y,
            7 + y,
        );
        let r: i16x8 = transmute(vaddw_high_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_high_s16() {
        let x = i32::MAX;
        let a = i32x4::new(x, 1, 2, 3);
        let y = i16::MAX;
        let b = i16x8::new(0, 0, 0, 0, y, y, y, y);
        let y = y as i32;
        let e = i32x4::new(x.wrapping_add(y), 1 + y, 2 + y, 3 + y);
        let r: i32x4 = transmute(vaddw_high_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_high_s32() {
        let x = i64::MAX;
        let a = i64x2::new(x, 1);
        let y = i32::MAX;
        let b = i32x4::new(0, 0, y, y);
        let y = y as i64;
        let e = i64x2::new(x.wrapping_add(y), 1 + y);
        let r: i64x2 = transmute(vaddw_high_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_high_u8() {
        let x = u16::MAX;
        let a = u16x8::new(x, 1, 2, 3, 4, 5, 6, 7);
        let y = u8::MAX;
        let b = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, y, y, y, y, y, y, y, y);
        let y = y as u16;
        let e = u16x8::new(
            x.wrapping_add(y),
            1 + y,
            2 + y,
            3 + y,
            4 + y,
            5 + y,
            6 + y,
            7 + y,
        );
        let r: u16x8 = transmute(vaddw_high_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_high_u16() {
        let x = u32::MAX;
        let a = u32x4::new(x, 1, 2, 3);
        let y = u16::MAX;
        let b = u16x8::new(0, 0, 0, 0, y, y, y, y);
        let y = y as u32;
        let e = u32x4::new(x.wrapping_add(y), 1 + y, 2 + y, 3 + y);
        let r: u32x4 = transmute(vaddw_high_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaddw_high_u32() {
        let x = u64::MAX;
        let a = u64x2::new(x, 1);
        let y = u32::MAX;
        let b = u32x4::new(0, 0, y, y);
        let y = y as u64;
        let e = u64x2::new(x.wrapping_add(y), 1 + y);
        let r: u64x2 = transmute(vaddw_high_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = i8x8::new(-1, -2, -3, -4, -5, -6, -7, -8);
        let r: i8x8 = transmute(vmvn_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = i8x16::new(
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        );
        let r: i8x16 = transmute(vmvnq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let e = i16x4::new(-1, -2, -3, -4);
        let r: i16x4 = transmute(vmvn_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = i16x8::new(-1, -2, -3, -4, -5, -6, -7, -8);
        let r: i16x8 = transmute(vmvnq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_s32() {
        let a = i32x2::new(0, 1);
        let e = i32x2::new(-1, -2);
        let r: i32x2 = transmute(vmvn_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let e = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vmvnq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = u8x8::new(255, 254, 253, 252, 251, 250, 249, 248);
        let r: u8x8 = transmute(vmvn_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = u8x16::new(
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
        );
        let r: u8x16 = transmute(vmvnq_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let e = u16x4::new(65_535, 65_534, 65_533, 65_532);
        let r: u16x4 = transmute(vmvn_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = u16x8::new(
            65_535, 65_534, 65_533, 65_532, 65_531, 65_530, 65_529, 65_528,
        );
        let r: u16x8 = transmute(vmvnq_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_u32() {
        let a = u32x2::new(0, 1);
        let e = u32x2::new(4_294_967_295, 4_294_967_294);
        let r: u32x2 = transmute(vmvn_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let e = u32x4::new(4_294_967_295, 4_294_967_294, 4_294_967_293, 4_294_967_292);
        let r: u32x4 = transmute(vmvnq_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvn_p8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e = u8x8::new(255, 254, 253, 252, 251, 250, 249, 248);
        let r: u8x8 = transmute(vmvn_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmvnq_p8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e = u8x16::new(
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
        );
        let r: u8x16 = transmute(vmvnq_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_s8() {
        let a = i8x8::new(0, -1, -2, -3, -4, -5, -6, -7);
        let b = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let e = i8x8::new(0, -2, -2, -4, -4, -6, -6, -8);
        let r: i8x8 = transmute(vbic_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_s8() {
        let a = i8x16::new(
            0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
        );
        let b = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let e = i8x16::new(
            0, -2, -2, -4, -4, -6, -6, -8, -8, -10, -10, -12, -12, -14, -14, -16,
        );
        let r: i8x16 = transmute(vbicq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_s16() {
        let a = i16x4::new(0, -1, -2, -3);
        let b = i16x4::new(1, 1, 1, 1);
        let e = i16x4::new(0, -2, -2, -4);
        let r: i16x4 = transmute(vbic_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_s16() {
        let a = i16x8::new(0, -1, -2, -3, -4, -5, -6, -7);
        let b = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let e = i16x8::new(0, -2, -2, -4, -4, -6, -6, -8);
        let r: i16x8 = transmute(vbicq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_s32() {
        let a = i32x2::new(0, -1);
        let b = i32x2::new(1, 1);
        let e = i32x2::new(0, -2);
        let r: i32x2 = transmute(vbic_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_s32() {
        let a = i32x4::new(0, -1, -2, -3);
        let b = i32x4::new(1, 1, 1, 1);
        let e = i32x4::new(0, -2, -2, -4);
        let r: i32x4 = transmute(vbicq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_s64() {
        let a = i64x1::new(-1);
        let b = i64x1::new(1);
        let e = i64x1::new(-2);
        let r: i64x1 = transmute(vbic_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_s64() {
        let a = i64x2::new(0, -1);
        let b = i64x2::new(1, 1);
        let e = i64x2::new(0, -2);
        let r: i64x2 = transmute(vbicq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let e = u8x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: u8x8 = transmute(vbic_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let e = u8x16::new(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
        let r: u8x16 = transmute(vbicq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let b = u16x4::new(1, 1, 1, 1);
        let e = u16x4::new(0, 0, 2, 2);
        let r: u16x4 = transmute(vbic_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let e = u16x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: u16x8 = transmute(vbicq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_u32() {
        let a = u32x2::new(0, 1);
        let b = u32x2::new(1, 1);
        let e = u32x2::new(0, 0);
        let r: u32x2 = transmute(vbic_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let b = u32x4::new(1, 1, 1, 1);
        let e = u32x4::new(0, 0, 2, 2);
        let r: u32x4 = transmute(vbicq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbic_u64() {
        let a = u64x1::new(1);
        let b = u64x1::new(1);
        let e = u64x1::new(0);
        let r: u64x1 = transmute(vbic_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbicq_u64() {
        let a = u64x2::new(0, 1);
        let b = u64x2::new(1, 1);
        let e = u64x2::new(0, 0);
        let r: u64x2 = transmute(vbicq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_s8() {
        let a = u8x8::new(u8::MAX, 1, u8::MAX, 2, u8::MAX, 0, u8::MAX, 0);
        let b = i8x8::new(
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
        );
        let c = i8x8::new(
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
        );
        let e = i8x8::new(
            i8::MAX,
            i8::MIN | 1,
            i8::MAX,
            i8::MIN | 2,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
        );
        let r: i8x8 = transmute(vbsl_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_s16() {
        let a = u16x4::new(u16::MAX, 0, 1, 2);
        let b = i16x4::new(i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        let c = i16x4::new(i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        let e = i16x4::new(i16::MAX, i16::MIN, i16::MIN | 1, i16::MIN | 2);
        let r: i16x4 = transmute(vbsl_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_s32() {
        let a = u32x2::new(u32::MAX, 1);
        let b = i32x2::new(i32::MAX, i32::MAX);
        let c = i32x2::new(i32::MIN, i32::MIN);
        let e = i32x2::new(i32::MAX, i32::MIN | 1);
        let r: i32x2 = transmute(vbsl_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_s64() {
        let a = u64x1::new(1);
        let b = i64x1::new(i64::MAX);
        let c = i64x1::new(i64::MIN);
        let e = i64x1::new(i64::MIN | 1);
        let r: i64x1 = transmute(vbsl_s64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_u8() {
        let a = u8x8::new(u8::MAX, 1, u8::MAX, 2, u8::MAX, 0, u8::MAX, 0);
        let b = u8x8::new(
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
        );
        let c = u8x8::new(
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
        );
        let e = u8x8::new(u8::MAX, 1, u8::MAX, 2, u8::MAX, u8::MIN, u8::MAX, u8::MIN);
        let r: u8x8 = transmute(vbsl_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_u16() {
        let a = u16x4::new(u16::MAX, 0, 1, 2);
        let b = u16x4::new(u16::MAX, u16::MAX, u16::MAX, u16::MAX);
        let c = u16x4::new(u16::MIN, u16::MIN, u16::MIN, u16::MIN);
        let e = u16x4::new(u16::MAX, 0, 1, 2);
        let r: u16x4 = transmute(vbsl_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_u32() {
        let a = u32x2::new(u32::MAX, 2);
        let b = u32x2::new(u32::MAX, u32::MAX);
        let c = u32x2::new(u32::MIN, u32::MIN);
        let e = u32x2::new(u32::MAX, 2);
        let r: u32x2 = transmute(vbsl_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_u64() {
        let a = u64x1::new(2);
        let b = u64x1::new(u64::MAX);
        let c = u64x1::new(u64::MIN);
        let e = u64x1::new(2);
        let r: u64x1 = transmute(vbsl_u64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_f32() {
        let a = u32x2::new(1, 0x80000000);
        let b = f32x2::new(8388609f32, -1.23f32);
        let c = f32x2::new(2097152f32, 2.34f32);
        let e = f32x2::new(2097152.25f32, -2.34f32);
        let r: f32x2 = transmute(vbsl_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_p8() {
        let a = u8x8::new(u8::MAX, 1, u8::MAX, 2, u8::MAX, 0, u8::MAX, 0);
        let b = u8x8::new(
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
        );
        let c = u8x8::new(
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
        );
        let e = u8x8::new(u8::MAX, 1, u8::MAX, 2, u8::MAX, u8::MIN, u8::MAX, u8::MIN);
        let r: u8x8 = transmute(vbsl_p8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbsl_p16() {
        let a = u16x4::new(u16::MAX, 0, 1, 2);
        let b = u16x4::new(u16::MAX, u16::MAX, u16::MAX, u16::MAX);
        let c = u16x4::new(u16::MIN, u16::MIN, u16::MIN, u16::MIN);
        let e = u16x4::new(u16::MAX, 0, 1, 2);
        let r: u16x4 = transmute(vbsl_p16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_s8() {
        let a = u8x16::new(
            u8::MAX,
            1,
            u8::MAX,
            2,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
        );
        let b = i8x16::new(
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
            i8::MAX,
        );
        let c = i8x16::new(
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
            i8::MIN,
        );
        let e = i8x16::new(
            i8::MAX,
            i8::MIN | 1,
            i8::MAX,
            i8::MIN | 2,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
        );
        let r: i8x16 = transmute(vbslq_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_s16() {
        let a = u16x8::new(u16::MAX, 1, u16::MAX, 2, u16::MAX, 0, u16::MAX, 0);
        let b = i16x8::new(
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MAX,
        );
        let c = i16x8::new(
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
            i16::MIN,
        );
        let e = i16x8::new(
            i16::MAX,
            i16::MIN | 1,
            i16::MAX,
            i16::MIN | 2,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
        );
        let r: i16x8 = transmute(vbslq_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_s32() {
        let a = u32x4::new(u32::MAX, 1, u32::MAX, 2);
        let b = i32x4::new(i32::MAX, i32::MAX, i32::MAX, i32::MAX);
        let c = i32x4::new(i32::MIN, i32::MIN, i32::MIN, i32::MIN);
        let e = i32x4::new(i32::MAX, i32::MIN | 1, i32::MAX, i32::MIN | 2);
        let r: i32x4 = transmute(vbslq_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_s64() {
        let a = u64x2::new(u64::MAX, 1);
        let b = i64x2::new(i64::MAX, i64::MAX);
        let c = i64x2::new(i64::MIN, i64::MIN);
        let e = i64x2::new(i64::MAX, i64::MIN | 1);
        let r: i64x2 = transmute(vbslq_s64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_u8() {
        let a = u8x16::new(
            u8::MAX,
            1,
            u8::MAX,
            2,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
        );
        let b = u8x16::new(
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
        );
        let c = u8x16::new(
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
        );
        let e = u8x16::new(
            u8::MAX,
            1,
            u8::MAX,
            2,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
        );
        let r: u8x16 = transmute(vbslq_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_u16() {
        let a = u16x8::new(u16::MAX, 1, u16::MAX, 2, u16::MAX, 0, u16::MAX, 0);
        let b = u16x8::new(
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
        );
        let c = u16x8::new(
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
        );
        let e = u16x8::new(
            u16::MAX,
            1,
            u16::MAX,
            2,
            u16::MAX,
            u16::MIN,
            u16::MAX,
            u16::MIN,
        );
        let r: u16x8 = transmute(vbslq_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_u32() {
        let a = u32x4::new(u32::MAX, 1, u32::MAX, 2);
        let b = u32x4::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
        let c = u32x4::new(u32::MIN, u32::MIN, u32::MIN, u32::MIN);
        let e = u32x4::new(u32::MAX, 1, u32::MAX, 2);
        let r: u32x4 = transmute(vbslq_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_u64() {
        let a = u64x2::new(u64::MAX, 1);
        let b = u64x2::new(u64::MAX, u64::MAX);
        let c = u64x2::new(u64::MIN, u64::MIN);
        let e = u64x2::new(u64::MAX, 1);
        let r: u64x2 = transmute(vbslq_u64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_f32() {
        let a = u32x4::new(u32::MAX, 0, 1, 0x80000000);
        let b = f32x4::new(-1.23f32, -1.23f32, 8388609f32, -1.23f32);
        let c = f32x4::new(2.34f32, 2.34f32, 2097152f32, 2.34f32);
        let e = f32x4::new(-1.23f32, 2.34f32, 2097152.25f32, -2.34f32);
        let r: f32x4 = transmute(vbslq_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_p8() {
        let a = u8x16::new(
            u8::MAX,
            1,
            u8::MAX,
            2,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
            u8::MAX,
            0,
        );
        let b = u8x16::new(
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
        );
        let c = u8x16::new(
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
            u8::MIN,
        );
        let e = u8x16::new(
            u8::MAX,
            1,
            u8::MAX,
            2,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
            u8::MAX,
            u8::MIN,
        );
        let r: u8x16 = transmute(vbslq_p8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vbslq_p16() {
        let a = u16x8::new(u16::MAX, 1, u16::MAX, 2, u16::MAX, 0, u16::MAX, 0);
        let b = u16x8::new(
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
        );
        let c = u16x8::new(
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
            u16::MIN,
        );
        let e = u16x8::new(
            u16::MAX,
            1,
            u16::MAX,
            2,
            u16::MAX,
            u16::MIN,
            u16::MAX,
            u16::MIN,
        );
        let r: u16x8 = transmute(vbslq_p16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_s8() {
        let a = i8x8::new(0, -1, -2, -3, -4, -5, -6, -7);
        let b = i8x8::new(-2, -2, -2, -2, -2, -2, -2, -2);
        let e = i8x8::new(1, -1, -1, -3, -3, -5, -5, -7);
        let r: i8x8 = transmute(vorn_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_s8() {
        let a = i8x16::new(
            0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15,
        );
        let b = i8x16::new(
            -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        );
        let e = i8x16::new(
            1, -1, -1, -3, -3, -5, -5, -7, -7, -9, -9, -11, -11, -13, -13, -15,
        );
        let r: i8x16 = transmute(vornq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_s16() {
        let a = i16x4::new(0, -1, -2, -3);
        let b = i16x4::new(-2, -2, -2, -2);
        let e = i16x4::new(1, -1, -1, -3);
        let r: i16x4 = transmute(vorn_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_s16() {
        let a = i16x8::new(0, -1, -2, -3, -4, -5, -6, -7);
        let b = i16x8::new(-2, -2, -2, -2, -2, -2, -2, -2);
        let e = i16x8::new(1, -1, -1, -3, -3, -5, -5, -7);
        let r: i16x8 = transmute(vornq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_s32() {
        let a = i32x2::new(0, -1);
        let b = i32x2::new(-2, -2);
        let e = i32x2::new(1, -1);
        let r: i32x2 = transmute(vorn_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_s32() {
        let a = i32x4::new(0, -1, -2, -3);
        let b = i32x4::new(-2, -2, -2, -2);
        let e = i32x4::new(1, -1, -1, -3);
        let r: i32x4 = transmute(vornq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_s64() {
        let a = i64x1::new(0);
        let b = i64x1::new(-2);
        let e = i64x1::new(1);
        let r: i64x1 = transmute(vorn_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_s64() {
        let a = i64x2::new(0, -1);
        let b = i64x2::new(-2, -2);
        let e = i64x2::new(1, -1);
        let r: i64x2 = transmute(vornq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let t = u8::MAX - 1;
        let b = u8x8::new(t, t, t, t, t, t, t, t);
        let e = u8x8::new(1, 1, 3, 3, 5, 5, 7, 7);
        let r: u8x8 = transmute(vorn_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let t = u8::MAX - 1;
        let b = u8x16::new(t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t);
        let e = u8x16::new(1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15);
        let r: u8x16 = transmute(vornq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let t = u16::MAX - 1;
        let b = u16x4::new(t, t, t, t);
        let e = u16x4::new(1, 1, 3, 3);
        let r: u16x4 = transmute(vorn_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let t = u16::MAX - 1;
        let b = u16x8::new(t, t, t, t, t, t, t, t);
        let e = u16x8::new(1, 1, 3, 3, 5, 5, 7, 7);
        let r: u16x8 = transmute(vornq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_u32() {
        let a = u32x2::new(0, 1);
        let t = u32::MAX - 1;
        let b = u32x2::new(t, t);
        let e = u32x2::new(1, 1);
        let r: u32x2 = transmute(vorn_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let t = u32::MAX - 1;
        let b = u32x4::new(t, t, t, t);
        let e = u32x4::new(1, 1, 3, 3);
        let r: u32x4 = transmute(vornq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorn_u64() {
        let a = u64x1::new(0);
        let t = u64::MAX - 1;
        let b = u64x1::new(t);
        let e = u64x1::new(1);
        let r: u64x1 = transmute(vorn_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vornq_u64() {
        let a = u64x2::new(0, 1);
        let t = u64::MAX - 1;
        let b = u64x2::new(t, t);
        let e = u64x2::new(1, 1);
        let r: u64x2 = transmute(vornq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vmovn_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let e = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vmovn_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_s64() {
        let a = i64x2::new(1, 2);
        let e = i32x2::new(1, 2);
        let r: i32x2 = transmute(vmovn_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vmovn_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let e = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vmovn_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovn_u64() {
        let a = u64x2::new(1, 2);
        let e = u32x2::new(1, 2);
        let r: u32x2 = transmute(vmovn_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_s8() {
        let e = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vmovl_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_s16() {
        let e = i32x4::new(1, 2, 3, 4);
        let a = i16x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vmovl_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_s32() {
        let e = i64x2::new(1, 2);
        let a = i32x2::new(1, 2);
        let r: i64x2 = transmute(vmovl_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_u8() {
        let e = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vmovl_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_u16() {
        let e = u32x4::new(1, 2, 3, 4);
        let a = u16x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vmovl_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmovl_u32() {
        let e = u64x2::new(1, 2);
        let a = u32x2::new(1, 2);
        let r: u64x2 = transmute(vmovl_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s8() {
        test_bit_s8(|i, j| vand_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s8() {
        testq_bit_s8(|i, j| vandq_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s16() {
        test_bit_s16(|i, j| vand_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s16() {
        testq_bit_s16(|i, j| vandq_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s32() {
        test_bit_s32(|i, j| vand_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s32() {
        testq_bit_s32(|i, j| vandq_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s64() {
        test_bit_s64(|i, j| vand_s64(i, j), |a: i64, b: i64| -> i64 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s64() {
        testq_bit_s64(|i, j| vandq_s64(i, j), |a: i64, b: i64| -> i64 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u8() {
        test_bit_u8(|i, j| vand_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u8() {
        testq_bit_u8(|i, j| vandq_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u16() {
        test_bit_u16(|i, j| vand_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u16() {
        testq_bit_u16(|i, j| vandq_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u32() {
        test_bit_u32(|i, j| vand_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u32() {
        testq_bit_u32(|i, j| vandq_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u64() {
        test_bit_u64(|i, j| vand_u64(i, j), |a: u64, b: u64| -> u64 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u64() {
        testq_bit_u64(|i, j| vandq_u64(i, j), |a: u64, b: u64| -> u64 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s8() {
        test_bit_s8(|i, j| vorr_s8(i, j), |a: i8, b: i8| -> i8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s8() {
        testq_bit_s8(|i, j| vorrq_s8(i, j), |a: i8, b: i8| -> i8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s16() {
        test_bit_s16(|i, j| vorr_s16(i, j), |a: i16, b: i16| -> i16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s16() {
        testq_bit_s16(|i, j| vorrq_s16(i, j), |a: i16, b: i16| -> i16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s32() {
        test_bit_s32(|i, j| vorr_s32(i, j), |a: i32, b: i32| -> i32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s32() {
        testq_bit_s32(|i, j| vorrq_s32(i, j), |a: i32, b: i32| -> i32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s64() {
        test_bit_s64(|i, j| vorr_s64(i, j), |a: i64, b: i64| -> i64 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s64() {
        testq_bit_s64(|i, j| vorrq_s64(i, j), |a: i64, b: i64| -> i64 { a | b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u8() {
        test_bit_u8(|i, j| vorr_u8(i, j), |a: u8, b: u8| -> u8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u8() {
        testq_bit_u8(|i, j| vorrq_u8(i, j), |a: u8, b: u8| -> u8 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u16() {
        test_bit_u16(|i, j| vorr_u16(i, j), |a: u16, b: u16| -> u16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u16() {
        testq_bit_u16(|i, j| vorrq_u16(i, j), |a: u16, b: u16| -> u16 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u32() {
        test_bit_u32(|i, j| vorr_u32(i, j), |a: u32, b: u32| -> u32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u32() {
        testq_bit_u32(|i, j| vorrq_u32(i, j), |a: u32, b: u32| -> u32 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u64() {
        test_bit_u64(|i, j| vorr_u64(i, j), |a: u64, b: u64| -> u64 { a | b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u64() {
        testq_bit_u64(|i, j| vorrq_u64(i, j), |a: u64, b: u64| -> u64 { a | b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s8() {
        test_bit_s8(|i, j| veor_s8(i, j), |a: i8, b: i8| -> i8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s8() {
        testq_bit_s8(|i, j| veorq_s8(i, j), |a: i8, b: i8| -> i8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s16() {
        test_bit_s16(|i, j| veor_s16(i, j), |a: i16, b: i16| -> i16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s16() {
        testq_bit_s16(|i, j| veorq_s16(i, j), |a: i16, b: i16| -> i16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s32() {
        test_bit_s32(|i, j| veor_s32(i, j), |a: i32, b: i32| -> i32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s32() {
        testq_bit_s32(|i, j| veorq_s32(i, j), |a: i32, b: i32| -> i32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s64() {
        test_bit_s64(|i, j| veor_s64(i, j), |a: i64, b: i64| -> i64 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s64() {
        testq_bit_s64(|i, j| veorq_s64(i, j), |a: i64, b: i64| -> i64 { a ^ b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u8() {
        test_bit_u8(|i, j| veor_u8(i, j), |a: u8, b: u8| -> u8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u8() {
        testq_bit_u8(|i, j| veorq_u8(i, j), |a: u8, b: u8| -> u8 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u16() {
        test_bit_u16(|i, j| veor_u16(i, j), |a: u16, b: u16| -> u16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u16() {
        testq_bit_u16(|i, j| veorq_u16(i, j), |a: u16, b: u16| -> u16 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u32() {
        test_bit_u32(|i, j| veor_u32(i, j), |a: u32, b: u32| -> u32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u32() {
        testq_bit_u32(|i, j| veorq_u32(i, j), |a: u32, b: u32| -> u32 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u64() {
        test_bit_u64(|i, j| veor_u64(i, j), |a: u64, b: u64| -> u64 { a ^ b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u64() {
        testq_bit_u64(|i, j| veorq_u64(i, j), |a: u64, b: u64| -> u64 { a ^ b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s8() {
        test_cmp_s8(
            |i, j| vceq_s8(i, j),
            |a: i8, b: i8| -> u8 { if a == b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s8() {
        testq_cmp_s8(
            |i, j| vceqq_s8(i, j),
            |a: i8, b: i8| -> u8 { if a == b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s16() {
        test_cmp_s16(
            |i, j| vceq_s16(i, j),
            |a: i16, b: i16| -> u16 { if a == b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s16() {
        testq_cmp_s16(
            |i, j| vceqq_s16(i, j),
            |a: i16, b: i16| -> u16 { if a == b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s32() {
        test_cmp_s32(
            |i, j| vceq_s32(i, j),
            |a: i32, b: i32| -> u32 { if a == b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s32() {
        testq_cmp_s32(
            |i, j| vceqq_s32(i, j),
            |a: i32, b: i32| -> u32 { if a == b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u8() {
        test_cmp_u8(
            |i, j| vceq_u8(i, j),
            |a: u8, b: u8| -> u8 { if a == b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u8() {
        testq_cmp_u8(
            |i, j| vceqq_u8(i, j),
            |a: u8, b: u8| -> u8 { if a == b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u16() {
        test_cmp_u16(
            |i, j| vceq_u16(i, j),
            |a: u16, b: u16| -> u16 { if a == b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u16() {
        testq_cmp_u16(
            |i, j| vceqq_u16(i, j),
            |a: u16, b: u16| -> u16 { if a == b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u32() {
        test_cmp_u32(
            |i, j| vceq_u32(i, j),
            |a: u32, b: u32| -> u32 { if a == b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u32() {
        testq_cmp_u32(
            |i, j| vceqq_u32(i, j),
            |a: u32, b: u32| -> u32 { if a == b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f32() {
        test_cmp_f32(
            |i, j| vcge_f32(i, j),
            |a: f32, b: f32| -> u32 { if a == b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f32() {
        testq_cmp_f32(
            |i, j| vcgeq_f32(i, j),
            |a: f32, b: f32| -> u32 { if a == b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s8() {
        test_cmp_s8(
            |i, j| vcgt_s8(i, j),
            |a: i8, b: i8| -> u8 { if a > b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s8() {
        testq_cmp_s8(
            |i, j| vcgtq_s8(i, j),
            |a: i8, b: i8| -> u8 { if a > b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s16() {
        test_cmp_s16(
            |i, j| vcgt_s16(i, j),
            |a: i16, b: i16| -> u16 { if a > b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s16() {
        testq_cmp_s16(
            |i, j| vcgtq_s16(i, j),
            |a: i16, b: i16| -> u16 { if a > b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s32() {
        test_cmp_s32(
            |i, j| vcgt_s32(i, j),
            |a: i32, b: i32| -> u32 { if a > b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s32() {
        testq_cmp_s32(
            |i, j| vcgtq_s32(i, j),
            |a: i32, b: i32| -> u32 { if a > b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u8() {
        test_cmp_u8(
            |i, j| vcgt_u8(i, j),
            |a: u8, b: u8| -> u8 { if a > b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u8() {
        testq_cmp_u8(
            |i, j| vcgtq_u8(i, j),
            |a: u8, b: u8| -> u8 { if a > b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u16() {
        test_cmp_u16(
            |i, j| vcgt_u16(i, j),
            |a: u16, b: u16| -> u16 { if a > b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u16() {
        testq_cmp_u16(
            |i, j| vcgtq_u16(i, j),
            |a: u16, b: u16| -> u16 { if a > b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u32() {
        test_cmp_u32(
            |i, j| vcgt_u32(i, j),
            |a: u32, b: u32| -> u32 { if a > b { 0xFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u32() {
        testq_cmp_u32(
            |i, j| vcgtq_u32(i, j),
            |a: u32, b: u32| -> u32 { if a > b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f32() {
        test_cmp_f32(
            |i, j| vcgt_f32(i, j),
            |a: f32, b: f32| -> u32 { if a > b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f32() {
        testq_cmp_f32(
            |i, j| vcgtq_f32(i, j),
            |a: f32, b: f32| -> u32 { if a > b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s8() {
        test_cmp_s8(
            |i, j| vclt_s8(i, j),
            |a: i8, b: i8| -> u8 { if a < b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s8() {
        testq_cmp_s8(
            |i, j| vcltq_s8(i, j),
            |a: i8, b: i8| -> u8 { if a < b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s16() {
        test_cmp_s16(
            |i, j| vclt_s16(i, j),
            |a: i16, b: i16| -> u16 { if a < b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s16() {
        testq_cmp_s16(
            |i, j| vcltq_s16(i, j),
            |a: i16, b: i16| -> u16 { if a < b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s32() {
        test_cmp_s32(
            |i, j| vclt_s32(i, j),
            |a: i32, b: i32| -> u32 { if a < b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s32() {
        testq_cmp_s32(
            |i, j| vcltq_s32(i, j),
            |a: i32, b: i32| -> u32 { if a < b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u8() {
        test_cmp_u8(
            |i, j| vclt_u8(i, j),
            |a: u8, b: u8| -> u8 { if a < b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u8() {
        testq_cmp_u8(
            |i, j| vcltq_u8(i, j),
            |a: u8, b: u8| -> u8 { if a < b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u16() {
        test_cmp_u16(
            |i, j| vclt_u16(i, j),
            |a: u16, b: u16| -> u16 { if a < b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u16() {
        testq_cmp_u16(
            |i, j| vcltq_u16(i, j),
            |a: u16, b: u16| -> u16 { if a < b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u32() {
        test_cmp_u32(
            |i, j| vclt_u32(i, j),
            |a: u32, b: u32| -> u32 { if a < b { 0xFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u32() {
        testq_cmp_u32(
            |i, j| vcltq_u32(i, j),
            |a: u32, b: u32| -> u32 { if a < b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f32() {
        test_cmp_f32(
            |i, j| vclt_f32(i, j),
            |a: f32, b: f32| -> u32 { if a < b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f32() {
        testq_cmp_f32(
            |i, j| vcltq_f32(i, j),
            |a: f32, b: f32| -> u32 { if a < b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s8() {
        test_cmp_s8(
            |i, j| vcle_s8(i, j),
            |a: i8, b: i8| -> u8 { if a <= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s8() {
        testq_cmp_s8(
            |i, j| vcleq_s8(i, j),
            |a: i8, b: i8| -> u8 { if a <= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s16() {
        test_cmp_s16(
            |i, j| vcle_s16(i, j),
            |a: i16, b: i16| -> u16 { if a <= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s16() {
        testq_cmp_s16(
            |i, j| vcleq_s16(i, j),
            |a: i16, b: i16| -> u16 { if a <= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s32() {
        test_cmp_s32(
            |i, j| vcle_s32(i, j),
            |a: i32, b: i32| -> u32 { if a <= b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s32() {
        testq_cmp_s32(
            |i, j| vcleq_s32(i, j),
            |a: i32, b: i32| -> u32 { if a <= b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u8() {
        test_cmp_u8(
            |i, j| vcle_u8(i, j),
            |a: u8, b: u8| -> u8 { if a <= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u8() {
        testq_cmp_u8(
            |i, j| vcleq_u8(i, j),
            |a: u8, b: u8| -> u8 { if a <= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u16() {
        test_cmp_u16(
            |i, j| vcle_u16(i, j),
            |a: u16, b: u16| -> u16 { if a <= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u16() {
        testq_cmp_u16(
            |i, j| vcleq_u16(i, j),
            |a: u16, b: u16| -> u16 { if a <= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u32() {
        test_cmp_u32(
            |i, j| vcle_u32(i, j),
            |a: u32, b: u32| -> u32 { if a <= b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u32() {
        testq_cmp_u32(
            |i, j| vcleq_u32(i, j),
            |a: u32, b: u32| -> u32 { if a <= b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f32() {
        test_cmp_f32(
            |i, j| vcle_f32(i, j),
            |a: f32, b: f32| -> u32 { if a <= b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f32() {
        testq_cmp_f32(
            |i, j| vcleq_f32(i, j),
            |a: f32, b: f32| -> u32 { if a <= b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s8() {
        test_cmp_s8(
            |i, j| vcge_s8(i, j),
            |a: i8, b: i8| -> u8 { if a >= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s8() {
        testq_cmp_s8(
            |i, j| vcgeq_s8(i, j),
            |a: i8, b: i8| -> u8 { if a >= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s16() {
        test_cmp_s16(
            |i, j| vcge_s16(i, j),
            |a: i16, b: i16| -> u16 { if a >= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s16() {
        testq_cmp_s16(
            |i, j| vcgeq_s16(i, j),
            |a: i16, b: i16| -> u16 { if a >= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s32() {
        test_cmp_s32(
            |i, j| vcge_s32(i, j),
            |a: i32, b: i32| -> u32 { if a >= b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s32() {
        testq_cmp_s32(
            |i, j| vcgeq_s32(i, j),
            |a: i32, b: i32| -> u32 { if a >= b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u8() {
        test_cmp_u8(
            |i, j| vcge_u8(i, j),
            |a: u8, b: u8| -> u8 { if a >= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u8() {
        testq_cmp_u8(
            |i, j| vcgeq_u8(i, j),
            |a: u8, b: u8| -> u8 { if a >= b { 0xFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u16() {
        test_cmp_u16(
            |i, j| vcge_u16(i, j),
            |a: u16, b: u16| -> u16 { if a >= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u16() {
        testq_cmp_u16(
            |i, j| vcgeq_u16(i, j),
            |a: u16, b: u16| -> u16 { if a >= b { 0xFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u32() {
        test_cmp_u32(
            |i, j| vcge_u32(i, j),
            |a: u32, b: u32| -> u32 { if a >= b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u32() {
        testq_cmp_u32(
            |i, j| vcgeq_u32(i, j),
            |a: u32, b: u32| -> u32 { if a >= b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f32() {
        test_cmp_f32(
            |i, j| vcge_f32(i, j),
            |a: f32, b: f32| -> u32 { if a >= b { 0xFFFFFFFF } else { 0 } },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f32() {
        testq_cmp_f32(
            |i, j| vcgeq_f32(i, j),
            |a: f32, b: f32| -> u32 { if a >= b { 0xFFFFFFFF } else { 0 } },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s8() {
        test_ari_s8(
            |i, j| vqsub_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s8() {
        testq_ari_s8(
            |i, j| vqsubq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s16() {
        test_ari_s16(
            |i, j| vqsub_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s16() {
        testq_ari_s16(
            |i, j| vqsubq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s32() {
        test_ari_s32(
            |i, j| vqsub_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s32() {
        testq_ari_s32(
            |i, j| vqsubq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_sub(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u8() {
        test_ari_u8(
            |i, j| vqsub_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u8() {
        testq_ari_u8(
            |i, j| vqsubq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u16() {
        test_ari_u16(
            |i, j| vqsub_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u16() {
        testq_ari_u16(
            |i, j| vqsubq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u32() {
        test_ari_u32(
            |i, j| vqsub_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_sub(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u32() {
        testq_ari_u32(
            |i, j| vqsubq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_sub(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s8() {
        test_ari_s8(|i, j| vhadd_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s8() {
        testq_ari_s8(|i, j| vhaddq_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s16() {
        test_ari_s16(|i, j| vhadd_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s16() {
        testq_ari_s16(|i, j| vhaddq_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s32() {
        test_ari_s32(|i, j| vhadd_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s32() {
        testq_ari_s32(|i, j| vhaddq_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u8() {
        test_ari_u8(|i, j| vhadd_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u8() {
        testq_ari_u8(|i, j| vhaddq_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u16() {
        test_ari_u16(|i, j| vhadd_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u16() {
        testq_ari_u16(|i, j| vhaddq_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u32() {
        test_ari_u32(|i, j| vhadd_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u32() {
        testq_ari_u32(|i, j| vhaddq_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s8() {
        test_ari_s8(|i, j| vrhadd_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s8() {
        testq_ari_s8(|i, j| vrhaddq_s8(i, j), |a: i8, b: i8| -> i8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s16() {
        test_ari_s16(|i, j| vrhadd_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s16() {
        testq_ari_s16(|i, j| vrhaddq_s16(i, j), |a: i16, b: i16| -> i16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s32() {
        test_ari_s32(|i, j| vrhadd_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s32() {
        testq_ari_s32(|i, j| vrhaddq_s32(i, j), |a: i32, b: i32| -> i32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u8() {
        test_ari_u8(|i, j| vrhadd_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u8() {
        testq_ari_u8(|i, j| vrhaddq_u8(i, j), |a: u8, b: u8| -> u8 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u16() {
        test_ari_u16(|i, j| vrhadd_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u16() {
        testq_ari_u16(|i, j| vrhaddq_u16(i, j), |a: u16, b: u16| -> u16 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u32() {
        test_ari_u32(|i, j| vrhadd_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u32() {
        testq_ari_u32(|i, j| vrhaddq_u32(i, j), |a: u32, b: u32| -> u32 { a & b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s8() {
        test_ari_s8(
            |i, j| vqadd_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s8() {
        testq_ari_s8(
            |i, j| vqaddq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s16() {
        test_ari_s16(
            |i, j| vqadd_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s16() {
        testq_ari_s16(
            |i, j| vqaddq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s32() {
        test_ari_s32(
            |i, j| vqadd_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s32() {
        testq_ari_s32(
            |i, j| vqaddq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.saturating_add(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u8() {
        test_ari_u8(
            |i, j| vqadd_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u8() {
        testq_ari_u8(
            |i, j| vqaddq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u16() {
        test_ari_u16(
            |i, j| vqadd_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u16() {
        testq_ari_u16(
            |i, j| vqaddq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u32() {
        test_ari_u32(
            |i, j| vqadd_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_add(b) },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u32() {
        testq_ari_u32(
            |i, j| vqaddq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.saturating_add(b) },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s8() {
        test_ari_s8(
            |i, j| vmul_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s8() {
        testq_ari_s8(
            |i, j| vmulq_s8(i, j),
            |a: i8, b: i8| -> i8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s16() {
        test_ari_s16(
            |i, j| vmul_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s16() {
        testq_ari_s16(
            |i, j| vmulq_s16(i, j),
            |a: i16, b: i16| -> i16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s32() {
        test_ari_s32(
            |i, j| vmul_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s32() {
        testq_ari_s32(
            |i, j| vmulq_s32(i, j),
            |a: i32, b: i32| -> i32 { a.overflowing_mul(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u8() {
        test_ari_u8(
            |i, j| vmul_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u8() {
        testq_ari_u8(
            |i, j| vmulq_u8(i, j),
            |a: u8, b: u8| -> u8 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u16() {
        test_ari_u16(
            |i, j| vmul_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u16() {
        testq_ari_u16(
            |i, j| vmulq_u16(i, j),
            |a: u16, b: u16| -> u16 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u32() {
        test_ari_u32(
            |i, j| vmul_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_mul(b).0 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u32() {
        testq_ari_u32(
            |i, j| vmulq_u32(i, j),
            |a: u32, b: u32| -> u32 { a.overflowing_mul(b).0 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f32() {
        test_ari_f32(|i, j| vmul_f32(i, j), |a: f32, b: f32| -> f32 { a * b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f32() {
        testq_ari_f32(|i, j| vmulq_f32(i, j), |a: f32, b: f32| -> f32 { a * b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s8() {
        test_ari_s8(|i, j| vsub_s8(i, j), |a: i8, b: i8| -> i8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s8() {
        testq_ari_s8(|i, j| vsubq_s8(i, j), |a: i8, b: i8| -> i8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s16() {
        test_ari_s16(|i, j| vsub_s16(i, j), |a: i16, b: i16| -> i16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s16() {
        testq_ari_s16(|i, j| vsubq_s16(i, j), |a: i16, b: i16| -> i16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s32() {
        test_ari_s32(|i, j| vsub_s32(i, j), |a: i32, b: i32| -> i32 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s32() {
        testq_ari_s32(|i, j| vsubq_s32(i, j), |a: i32, b: i32| -> i32 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u8() {
        test_ari_u8(|i, j| vsub_u8(i, j), |a: u8, b: u8| -> u8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u8() {
        testq_ari_u8(|i, j| vsubq_u8(i, j), |a: u8, b: u8| -> u8 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u16() {
        test_ari_u16(|i, j| vsub_u16(i, j), |a: u16, b: u16| -> u16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u16() {
        testq_ari_u16(|i, j| vsubq_u16(i, j), |a: u16, b: u16| -> u16 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u32() {
        test_ari_u32(|i, j| vsub_u32(i, j), |a: u32, b: u32| -> u32 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u32() {
        testq_ari_u32(|i, j| vsubq_u32(i, j), |a: u32, b: u32| -> u32 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f32() {
        test_ari_f32(|i, j| vsub_f32(i, j), |a: f32, b: f32| -> f32 { a - b });
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f32() {
        testq_ari_f32(|i, j| vsubq_f32(i, j), |a: f32, b: f32| -> f32 { a - b });
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s8() {
        test_ari_s8(
            |i, j| vhsub_s8(i, j),
            |a: i8, b: i8| -> i8 { (((a as i16) - (b as i16)) / 2) as i8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s8() {
        testq_ari_s8(
            |i, j| vhsubq_s8(i, j),
            |a: i8, b: i8| -> i8 { (((a as i16) - (b as i16)) / 2) as i8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s16() {
        test_ari_s16(
            |i, j| vhsub_s16(i, j),
            |a: i16, b: i16| -> i16 { (((a as i32) - (b as i32)) / 2) as i16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s16() {
        testq_ari_s16(
            |i, j| vhsubq_s16(i, j),
            |a: i16, b: i16| -> i16 { (((a as i32) - (b as i32)) / 2) as i16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s32() {
        test_ari_s32(
            |i, j| vhsub_s32(i, j),
            |a: i32, b: i32| -> i32 { (((a as i64) - (b as i64)) / 2) as i32 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s32() {
        testq_ari_s32(
            |i, j| vhsubq_s32(i, j),
            |a: i32, b: i32| -> i32 { (((a as i64) - (b as i64)) / 2) as i32 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u8() {
        test_ari_u8(
            |i, j| vhsub_u8(i, j),
            |a: u8, b: u8| -> u8 { (((a as u16) - (b as u16)) / 2) as u8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u8() {
        testq_ari_u8(
            |i, j| vhsubq_u8(i, j),
            |a: u8, b: u8| -> u8 { (((a as u16) - (b as u16)) / 2) as u8 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u16() {
        test_ari_u16(
            |i, j| vhsub_u16(i, j),
            |a: u16, b: u16| -> u16 { (((a as u16) - (b as u16)) / 2) as u16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u16() {
        testq_ari_u16(
            |i, j| vhsubq_u16(i, j),
            |a: u16, b: u16| -> u16 { (((a as u16) - (b as u16)) / 2) as u16 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u32() {
        test_ari_u32(
            |i, j| vhsub_u32(i, j),
            |a: u32, b: u32| -> u32 { (((a as u64) - (b as u64)) / 2) as u32 },
        );
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u32() {
        testq_ari_u32(
            |i, j| vhsubq_u32(i, j),
            |a: u32, b: u32| -> u32 { (((a as u64) - (b as u64)) / 2) as u32 },
        );
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vaba_s8() {
        let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let c = i8x8::new(10, 9, 8, 7, 6, 5, 4, 3);
        let r: i8x8 = transmute(vaba_s8(transmute(a), transmute(b), transmute(c)));
        let e = i8x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaba_s16() {
        let a = i16x4::new(1, 2, 3, 4);
        let b = i16x4::new(1, 1, 1, 1);
        let c = i16x4::new(10, 9, 8, 7);
        let r: i16x4 = transmute(vaba_s16(transmute(a), transmute(b), transmute(c)));
        let e = i16x4::new(10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaba_s32() {
        let a = i32x2::new(1, 2);
        let b = i32x2::new(1, 1);
        let c = i32x2::new(10, 9);
        let r: i32x2 = transmute(vaba_s32(transmute(a), transmute(b), transmute(c)));
        let e = i32x2::new(10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaba_u8() {
        let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let c = u8x8::new(10, 9, 8, 7, 6, 5, 4, 3);
        let r: u8x8 = transmute(vaba_u8(transmute(a), transmute(b), transmute(c)));
        let e = u8x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaba_u16() {
        let a = u16x4::new(1, 2, 3, 4);
        let b = u16x4::new(1, 1, 1, 1);
        let c = u16x4::new(10, 9, 8, 7);
        let r: u16x4 = transmute(vaba_u16(transmute(a), transmute(b), transmute(c)));
        let e = u16x4::new(10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vaba_u32() {
        let a = u32x2::new(1, 2);
        let b = u32x2::new(1, 1);
        let c = u32x2::new(10, 9);
        let r: u32x2 = transmute(vaba_u32(transmute(a), transmute(b), transmute(c)));
        let e = u32x2::new(10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabaq_s8() {
        let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2);
        let b = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let c = i8x16::new(10, 9, 8, 7, 6, 5, 4, 3, 12, 13, 14, 15, 16, 17, 18, 19);
        let r: i8x16 = transmute(vabaq_s8(transmute(a), transmute(b), transmute(c)));
        let e = i8x16::new(
            10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20,
        );
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabaq_s16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let c = i16x8::new(10, 9, 8, 7, 6, 5, 4, 3);
        let r: i16x8 = transmute(vabaq_s16(transmute(a), transmute(b), transmute(c)));
        let e = i16x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabaq_s32() {
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(1, 1, 1, 1);
        let c = i32x4::new(10, 9, 8, 7);
        let r: i32x4 = transmute(vabaq_s32(transmute(a), transmute(b), transmute(c)));
        let e = i32x4::new(10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabaq_u8() {
        let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2);
        let b = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let c = u8x16::new(10, 9, 8, 7, 6, 5, 4, 3, 12, 13, 14, 15, 16, 17, 18, 19);
        let r: u8x16 = transmute(vabaq_u8(transmute(a), transmute(b), transmute(c)));
        let e = u8x16::new(
            10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20,
        );
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabaq_u16() {
        let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = u16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let c = u16x8::new(10, 9, 8, 7, 6, 5, 4, 3);
        let r: u16x8 = transmute(vabaq_u16(transmute(a), transmute(b), transmute(c)));
        let e = u16x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vabaq_u32() {
        let a = u32x4::new(1, 2, 3, 4);
        let b = u32x4::new(1, 1, 1, 1);
        let c = u32x4::new(10, 9, 8, 7);
        let r: u32x4 = transmute(vabaq_u32(transmute(a), transmute(b), transmute(c)));
        let e = u32x4::new(10, 10, 10, 10);
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrev16_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i8x8::new(1, 0, 3, 2, 5, 4, 7, 6);
        let e: i8x8 = transmute(vrev16_s8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev16q_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = i8x16::new(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        let e: i8x16 = transmute(vrev16q_s8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev16_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u8x8::new(1, 0, 3, 2, 5, 4, 7, 6);
        let e: u8x8 = transmute(vrev16_u8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev16q_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = u8x16::new(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        let e: u8x16 = transmute(vrev16q_u8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev16_p8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i8x8::new(1, 0, 3, 2, 5, 4, 7, 6);
        let e: i8x8 = transmute(vrev16_p8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev16q_p8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = u8x16::new(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        let e: u8x16 = transmute(vrev16q_p8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i8x8::new(3, 2, 1, 0, 7, 6, 5, 4);
        let e: i8x8 = transmute(vrev32_s8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32q_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = i8x16::new(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        let e: i8x16 = transmute(vrev32q_s8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u8x8::new(3, 2, 1, 0, 7, 6, 5, 4);
        let e: u8x8 = transmute(vrev32_u8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32q_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = u8x16::new(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        let e: u8x16 = transmute(vrev32q_u8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let r = i16x4::new(1, 0, 3, 2);
        let e: i16x4 = transmute(vrev32_s16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32q_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i16x8::new(1, 0, 3, 2, 5, 4, 7, 6);
        let e: i16x8 = transmute(vrev32q_s16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32_p16() {
        let a = i16x4::new(0, 1, 2, 3);
        let r = i16x4::new(1, 0, 3, 2);
        let e: i16x4 = transmute(vrev32_p16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32q_p16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i16x8::new(1, 0, 3, 2, 5, 4, 7, 6);
        let e: i16x8 = transmute(vrev32q_p16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let r = u16x4::new(1, 0, 3, 2);
        let e: u16x4 = transmute(vrev32_u16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32q_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u16x8::new(1, 0, 3, 2, 5, 4, 7, 6);
        let e: u16x8 = transmute(vrev32q_u16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32_p8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u8x8::new(3, 2, 1, 0, 7, 6, 5, 4);
        let e: u8x8 = transmute(vrev32_p8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev32q_p8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = u8x16::new(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        let e: u8x16 = transmute(vrev32q_p8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_s8() {
        let a = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i8x8::new(7, 6, 5, 4, 3, 2, 1, 0);
        let e: i8x8 = transmute(vrev64_s8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_s8() {
        let a = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = i8x16::new(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
        let e: i8x16 = transmute(vrev64q_s8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_s16() {
        let a = i16x4::new(0, 1, 2, 3);
        let r = i16x4::new(3, 2, 1, 0);
        let e: i16x4 = transmute(vrev64_s16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_s16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = i16x8::new(3, 2, 1, 0, 7, 6, 5, 4);
        let e: i16x8 = transmute(vrev64q_s16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_s32() {
        let a = i32x2::new(0, 1);
        let r = i32x2::new(1, 0);
        let e: i32x2 = transmute(vrev64_s32(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_s32() {
        let a = i32x4::new(0, 1, 2, 3);
        let r = i32x4::new(1, 0, 3, 2);
        let e: i32x4 = transmute(vrev64q_s32(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_u8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u8x8::new(7, 6, 5, 4, 3, 2, 1, 0);
        let e: u8x8 = transmute(vrev64_u8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_u8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = u8x16::new(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
        let e: u8x16 = transmute(vrev64q_u8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_u16() {
        let a = u16x4::new(0, 1, 2, 3);
        let r = u16x4::new(3, 2, 1, 0);
        let e: u16x4 = transmute(vrev64_u16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_u16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u16x8::new(3, 2, 1, 0, 7, 6, 5, 4);
        let e: u16x8 = transmute(vrev64q_u16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_u32() {
        let a = u32x2::new(0, 1);
        let r = u32x2::new(1, 0);
        let e: u32x2 = transmute(vrev64_u32(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_u32() {
        let a = u32x4::new(0, 1, 2, 3);
        let r = u32x4::new(1, 0, 3, 2);
        let e: u32x4 = transmute(vrev64q_u32(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_f32() {
        let a = f32x2::new(1.0, 2.0);
        let r = f32x2::new(2.0, 1.0);
        let e: f32x2 = transmute(vrev64_f32(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_f32() {
        let a = f32x4::new(1.0, 2.0, -2.0, -1.0);
        let r = f32x4::new(2.0, 1.0, -1.0, -2.0);
        let e: f32x4 = transmute(vrev64q_f32(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_p8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u8x8::new(7, 6, 5, 4, 3, 2, 1, 0);
        let e: u8x8 = transmute(vrev64_p8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_p8() {
        let a = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = u8x16::new(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
        let e: u8x16 = transmute(vrev64q_p8(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64_p16() {
        let a = u16x4::new(0, 1, 2, 3);
        let r = u16x4::new(3, 2, 1, 0);
        let e: u16x4 = transmute(vrev64_p16(transmute(a)));
        assert_eq!(r, e);
    }
    #[simd_test(enable = "neon")]
    unsafe fn test_vrev64q_p16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r = u16x8::new(3, 2, 1, 0, 7, 6, 5, 4);
        let e: u16x8 = transmute(vrev64q_p16(transmute(a)));
        assert_eq!(r, e);
    }

    macro_rules! test_vcombine {
        ($test_id:ident => $fn_id:ident ([$($a:expr),*], [$($b:expr),*])) => {
            #[allow(unused_assignments)]
            #[simd_test(enable = "neon")]
            unsafe fn $test_id() {
                let a = [$($a),*];
                let b = [$($b),*];
                let e = [$($a),* $(, $b)*];
                let c = $fn_id(transmute(a), transmute(b));
                let mut d = e;
                d = transmute(c);
                assert_eq!(d, e);
            }
        }
    }

    test_vcombine!(test_vcombine_s8 => vcombine_s8([3_i8, -4, 5, -6, 7, 8, 9, 10], [13_i8, -14, 15, -16, 17, 18, 19, 110]));
    test_vcombine!(test_vcombine_u8 => vcombine_u8([3_u8, 4, 5, 6, 7, 8, 9, 10], [13_u8, 14, 15, 16, 17, 18, 19, 110]));
    test_vcombine!(test_vcombine_p8 => vcombine_p8([3_u8, 4, 5, 6, 7, 8, 9, 10], [13_u8, 14, 15, 16, 17, 18, 19, 110]));

    test_vcombine!(test_vcombine_s16 => vcombine_s16([3_i16, -4, 5, -6], [13_i16, -14, 15, -16]));
    test_vcombine!(test_vcombine_u16 => vcombine_u16([3_u16, 4, 5, 6], [13_u16, 14, 15, 16]));
    test_vcombine!(test_vcombine_p16 => vcombine_p16([3_u16, 4, 5, 6], [13_u16, 14, 15, 16]));
    test_vcombine!(test_vcombine_f16 => vcombine_f16([3_f16, 4., 5., 6.],
    [13_f16, 14., 15., 16.]));

    test_vcombine!(test_vcombine_s32 => vcombine_s32([3_i32, -4], [13_i32, -14]));
    test_vcombine!(test_vcombine_u32 => vcombine_u32([3_u32, 4], [13_u32, 14]));
    // note: poly32x4 does not exist, and neither does vcombine_p32
    test_vcombine!(test_vcombine_f32 => vcombine_f32([3_f32, -4.], [13_f32, -14.]));

    test_vcombine!(test_vcombine_s64 => vcombine_s64([-3_i64], [13_i64]));
    test_vcombine!(test_vcombine_u64 => vcombine_u64([3_u64], [13_u64]));
    test_vcombine!(test_vcombine_p64 => vcombine_p64([3_u64], [13_u64]));
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    test_vcombine!(test_vcombine_f64 => vcombine_f64([-3_f64], [13_f64]));
}

#[cfg(all(test, target_arch = "arm"))]
mod table_lookup_tests;

#[cfg(all(test, target_arch = "arm"))]
mod shift_and_insert_tests;

#[cfg(all(test, target_arch = "arm"))]
mod load_tests;

#[cfg(all(test, target_arch = "arm"))]
mod store_tests;
