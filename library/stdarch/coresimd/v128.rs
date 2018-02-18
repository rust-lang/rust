//! 128-bit wide vector types

use prelude::v1::*;

use coresimd::simd_llvm::*;

define_ty! { f64x2, f64, f64 }
define_impl! { f64x2, f64, 2, i64x2, x0, x1 }

define_ty! { f32x4, f32, f32, f32, f32 }
define_impl! { f32x4, f32, 4, i32x4, x0, x1, x2, x3 }

define_ty! { u64x2, u64, u64 }
define_impl! { u64x2, u64, 2, i64x2, x0, x1 }

define_ty! { i64x2, i64, i64 }
define_impl! { i64x2, i64, 2, i64x2, x0, x1 }

define_ty! { u32x4, u32, u32, u32, u32 }
define_impl! { u32x4, u32, 4, i32x4, x0, x1, x2, x3 }

define_ty! { i32x4, i32, i32, i32, i32 }
define_impl! { i32x4, i32, 4, i32x4, x0, x1, x2, x3 }

define_ty! { u16x8, u16, u16, u16, u16, u16, u16, u16, u16 }
define_impl! { u16x8, u16, 8, i16x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! { i16x8, i16, i16, i16, i16, i16, i16, i16, i16 }
define_impl! { i16x8, i16, 8, i16x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! {
    u8x16, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8
}
define_impl! {
    u8x16, u8, 16, i8x16,
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty! {
    i8x16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
}
define_impl! {
    i8x16, i8, 16, i8x16,
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
}

define_from!(u64x2, i64x2, u32x4, i32x4, u16x8, i16x8, u8x16, i8x16);
define_from!(i64x2, u64x2, u32x4, i32x4, u16x8, i16x8, u8x16, i8x16);
define_from!(u32x4, u64x2, i64x2, i32x4, u16x8, i16x8, u8x16, i8x16);
define_from!(i32x4, u64x2, i64x2, u32x4, u16x8, i16x8, u8x16, i8x16);
define_from!(u16x8, u64x2, i64x2, u32x4, i32x4, i16x8, u8x16, i8x16);
define_from!(i16x8, u64x2, i64x2, u32x4, i32x4, u16x8, u8x16, i8x16);
define_from!(u8x16, u64x2, i64x2, u32x4, i32x4, u16x8, i16x8, i8x16);
define_from!(i8x16, u64x2, i64x2, u32x4, i32x4, u16x8, i16x8, u8x16);

define_common_ops!(
    f64x2,
    f32x4,
    u64x2,
    i64x2,
    u32x4,
    i32x4,
    u16x8,
    i16x8,
    u8x16,
    i8x16
);
define_float_ops!(f64x2, f32x4);
define_integer_ops!(
    (u64x2, u64),
    (i64x2, i64),
    (u32x4, u32),
    (i32x4, i32),
    (u16x8, u16),
    (i16x8, i16),
    (u8x16, u8),
    (i8x16, i8)
);
define_signed_integer_ops!(i64x2, i32x4, i16x8, i8x16);
define_casts!(
    (f64x2, f32x2, as_f32x2),
    (f64x2, u64x2, as_u64x2),
    (f64x2, i64x2, as_i64x2),
    (f32x4, f64x4, as_f64x4),
    (f32x4, u32x4, as_u32x4),
    (f32x4, i32x4, as_i32x4),
    (u64x2, f64x2, as_f64x2),
    (u64x2, i64x2, as_i64x2),
    (i64x2, f64x2, as_f64x2),
    (i64x2, u64x2, as_u64x2),
    (u32x4, f32x4, as_f32x4),
    (u32x4, i32x4, as_i32x4),
    (i32x4, f32x4, as_f32x4),
    (i32x4, u32x4, as_u32x4),
    (u16x8, i16x8, as_i16x8),
    (i16x8, u16x8, as_u16x8),
    (u8x16, i8x16, as_i8x16),
    (i8x16, u8x16, as_u8x16)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operators() {
        test_ops_si!(i8x16, i16x8, i32x4, i64x2);
        test_ops_ui!(u8x16, u16x8, u32x4, u64x2);
        test_ops_f!(f32x4, f64x2);
    }
}
