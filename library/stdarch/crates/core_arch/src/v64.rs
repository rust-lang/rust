//! 64-bit wide vector types

use crate::prelude::v1::*;

use crate::core_arch::simd_llvm::*;

define_ty_doc! {
    f32x2, f32, f32 |
    /// A 64-bit vector with 2 `f32` lanes.
}
define_impl! { f32x2, f32, 2, i32x2, x0, x1 }

define_ty_doc! {
    u32x2, u32, u32 |
    /// A 64-bit vector with 2 `u32` lanes.
}
define_impl! { u32x2, u32, 2, i32x2, x0, x1 }

define_ty! { i32x2, i32, i32 }
define_impl! { i32x2, i32, 2, i32x2, x0, x1 }

define_ty! { u16x4, u16, u16, u16, u16 }
define_impl! { u16x4, u16, 4, i16x4, x0, x1, x2, x3 }

define_ty! { i16x4, i16, i16, i16, i16 }
define_impl! { i16x4, i16, 4, i16x4, x0, x1, x2, x3 }

define_ty! { u8x8, u8, u8, u8, u8, u8, u8, u8, u8 }
define_impl! { u8x8, u8, 8, i8x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! { i8x8, i8, i8, i8, i8, i8, i8, i8, i8 }
define_impl! { i8x8, i8, 8, i8x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_from!(u32x2, i32x2, u16x4, i16x4, u8x8, i8x8);
define_from!(i32x2, u32x2, u16x4, i16x4, u8x8, i8x8);
define_from!(u16x4, u32x2, i32x2, i16x4, u8x8, i8x8);
define_from!(i16x4, u32x2, i32x2, u16x4, u8x8, i8x8);
define_from!(u8x8, u32x2, i32x2, u16x4, i16x4, i8x8);
define_from!(i8x8, u32x2, i32x2, u16x4, i16x4, u8x8);

define_common_ops!(f32x2, u32x2, i32x2, u16x4, i16x4, u8x8, i8x8);
define_float_ops!(f32x2);
define_integer_ops!(
    (u32x2, u32),
    (i32x2, i32),
    (u16x4, u16),
    (i16x4, i16),
    (u8x8, u8),
    (i8x8, i8)
);
define_signed_integer_ops!(i32x2, i16x4, i8x8);
define_casts!(
    (f32x2, f64x2, as_f64x2),
    (f32x2, u32x2, as_u32x2),
    (f32x2, i32x2, as_i32x2),
    (u32x2, f32x2, as_f32x2),
    (u32x2, i32x2, as_i32x2),
    (i32x2, f32x2, as_f32x2),
    (i32x2, u32x2, as_u32x2),
    (u16x4, i16x4, as_i16x4),
    (i16x4, u16x4, as_u16x4),
    (u8x8, i8x8, as_i8x8),
    (i8x8, u8x8, as_u8x8),
    (i8x8, i16x8, as_i16x8),
    (u8x8, i16x8, as_i16x8),
    (i16x4, i32x4, as_i32x4),
    (i32x2, i64x2, as_i64x2),
    (u8x8, u16x8, as_u16x8),
    (u16x4, u32x4, as_u32x4),
    (u16x4, i32x4, as_i32x4),
    (u32x2, u64x2, as_u64x2),
    (u32x2, i64x2, as_i64x2)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators() {
        test_ops_si!(i8x8, i16x4, i32x2);
        test_ops_ui!(u8x8, u16x4, u32x2);
        test_ops_f!(f32x2);
    }
}
