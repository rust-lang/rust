use simd::*;

define_ty! { f64x4, f64, f64, f64, f64 }
define_impl! { f64x4, f64, 4, i64x4, x0, x1, x2, x3 }

define_ty! { f32x8, f32, f32, f32, f32, f32, f32, f32, f32 }
define_impl! { f32x8, f32, 8, i32x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! { u64x4, u64, u64, u64, u64 }
define_impl! { u64x4, u64, 4, i64x4, x0, x1, x2, x3 }

define_ty! { i64x4, i64, i64, i64, i64 }
define_impl! { i64x4, i64, 4, i64x4, x0, x1, x2, x3 }

define_ty! { u32x8, u32, u32, u32, u32, u32, u32, u32, u32 }
define_impl! { u32x8, u32, 8, i32x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! { i32x8, i32, i32, i32, i32, i32, i32, i32, i32 }
define_impl! { i32x8, i32, 8, i32x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! {
    u16x16,
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16
}
define_impl! {
    u16x16, u16, 16, i16x16,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty! {
    i16x16,
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16
}
define_impl! {
    i16x16, i16, 16, i16x16,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty! {
    u8x32,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8
}
define_impl! {
    u8x32, u8, 32, i8x32,
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31
}

define_ty! {
    i8x32,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
}
define_impl! {
    i8x32, i8, 32, i8x32,
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31
}

define_from!(u64x4, i64x4, u32x8, i32x8, u16x16, i16x16, u8x32, i8x32);
define_from!(i64x4, u64x4, u32x8, i32x8, u16x16, i16x16, u8x32, i8x32);
define_from!(u32x8, u64x4, i64x4, i32x8, u16x16, i16x16, u8x32, i8x32);
define_from!(i32x8, u64x4, i64x4, u32x8, u16x16, i16x16, u8x32, i8x32);
define_from!(u16x16, u64x4, i64x4, u32x8, i32x8, i16x16, u8x32, i8x32);
define_from!(i16x16, u64x4, i64x4, u32x8, i32x8, u16x16, u8x32, i8x32);
define_from!(u8x32, u64x4, i64x4, u32x8, i32x8, u16x16, i16x16, i8x32);
define_from!(i8x32, u64x4, i64x4, u32x8, i32x8, u16x16, i16x16, u8x32);

define_common_ops!(
    f64x4, f32x8, u64x4, i64x4, u32x8, i32x8, u16x16, i16x16, u8x32, i8x32);
define_float_ops!(f64x4, f32x8);
define_integer_ops!(
    (u64x4, u64),
    (i64x4, i64),
    (u32x8, u32),
    (i32x8, i32),
    (u16x16, u16),
    (i16x16, i16),
    (u8x32, u8),
    (i8x32, i8));
define_casts!(
    (f64x4, f32x4, as_f32x4),
    (f64x4, u64x4, as_u64x4),
    (f64x4, i64x4, as_i64x4),
    (f32x8, u32x8, as_u32x8),
    (f32x8, i32x8, as_i32x8),
    (u64x4, f64x4, as_f64x4),
    (u64x4, i64x4, as_i64x4),
    (i64x4, f64x4, as_f64x4),
    (i64x4, u64x4, as_u64x4),
    (u32x8, f32x8, as_f32x8),
    (u32x8, i32x8, as_i32x8),
    (i32x8, f32x8, as_f32x8),
    (i32x8, u32x8, as_u32x8),
    (u16x16, i16x16, as_i16x16),
    (i16x16, u16x16, as_u16x16),
    (u8x32, i8x32, as_i8x32),
    (i8x32, u8x32, as_u8x32));
