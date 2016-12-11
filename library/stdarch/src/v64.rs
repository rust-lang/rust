use simd::*;

define_ty! { f32x2, f32, f32 }
define_impl! { f32x2, f32, 2, x0, x1 }

define_ty! { u32x2, u32, u32 }
define_impl! { u32x2, u32, 2, x0, x1 }

define_ty! { i32x2, i32, i32 }
define_impl! { i32x2, i32, 2, x0, x1 }

define_ty! { u16x4, u16, u16, u16, u16 }
define_impl! { u16x4, u16, 4, x0, x1, x2, x3 }

define_ty! { i16x4, i16, i16, i16, i16 }
define_impl! { i16x4, i16, 4, x0, x1, x2, x3 }

define_ty! { u8x8, u8, u8, u8, u8, u8, u8, u8, u8 }
define_impl! { u8x8, u8, 8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! { i8x8, i8, i8, i8, i8, i8, i8, i8, i8 }
define_impl! { i8x8, i8, 8, x0, x1, x2, x3, x4, x5, x6, x7 }

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
    (i8x8, i8));
