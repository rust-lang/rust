//! 512-bit wide vector types

use simd_llvm::*;

define_ty! { f64x8, f64, f64, f64, f64, f64, f64, f64, f64 }
define_impl! { f64x8, f64, 8, i64x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! {
    f32x16,
    f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32
}
define_impl! {
    f32x16, f32, 16, i32x16,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty! { u64x8, u64, u64, u64, u64, u64, u64, u64, u64 }
define_impl! { u64x8, u64, 8, i64x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! { i64x8, i64, i64, i64, i64, i64, i64, i64, i64 }
define_impl! { i64x8, i64, 8, i64x8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty! {
    u32x16,
    u32, u32, u32, u32, u32, u32, u32, u32,
    u32, u32, u32, u32, u32, u32, u32, u32
}
define_impl! {
    u32x16, u32, 16, i32x16,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty! {
    i32x16,
    i32, i32, i32, i32, i32, i32, i32, i32,
    i32, i32, i32, i32, i32, i32, i32, i32
}
define_impl! {
    i32x16, i32, 16, i32x16,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty! {
    u16x32,
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16,
    u16, u16, u16, u16, u16, u16, u16, u16
}
define_impl! {
    u16x32, u16, 32, i16x32,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31
}

define_ty! {
    i16x32,
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16,
    i16, i16, i16, i16, i16, i16, i16, i16
}
define_impl! {
    i16x32, i16, 32, i16x32,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31
}

define_ty! {
    u8x64,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8,
    u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8
}
define_impl! {
    u8x64, u8, 64, i8x64,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,
    x32, x33, x34, x35, x36, x37, x38, x39,
    x40, x41, x42, x43, x44, x45, x46, x47,
    x48, x49, x50, x51, x52, x53, x54, x55,
    x56, x57, x58, x59, x60, x61, x62, x63
}

define_ty! {
    i8x64,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,
    i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
}
define_impl! {
    i8x64, i8, 64, i8x64,
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,
    x32, x33, x34, x35, x36, x37, x38, x39,
    x40, x41, x42, x43, x44, x45, x46, x47,
    x48, x49, x50, x51, x52, x53, x54, x55,
    x56, x57, x58, x59, x60, x61, x62, x63
}

define_from!(u64x8, i64x8, u32x16, i32x16, u16x32, i16x32, u8x64, i8x64);
define_from!(i64x8, u64x8, u32x16, i32x16, u16x32, i16x32, u8x64, i8x64);
define_from!(u32x16, u64x8, i64x8, i32x16, u16x32, i16x32, u8x64, i8x64);
define_from!(i32x16, u64x8, i64x8, u32x16, u16x32, i16x32, u8x64, i8x64);
define_from!(u16x32, u64x8, i64x8, u32x16, i32x16, i16x32, u8x64, i8x64);
define_from!(i16x32, u64x8, i64x8, u32x16, i32x16, u16x32, u8x64, i8x64);
define_from!(u8x64, u64x8, i64x8, u32x16, i32x16, u16x32, i16x32, i8x64);
define_from!(i8x64, u64x8, i64x8, u32x16, i32x16, u16x32, i16x32, u8x64);

define_common_ops!(
    f64x8,
    f32x16,
    u64x8,
    i64x8,
    u32x16,
    i32x16,
    u16x32,
    i16x32,
    u8x64,
    i8x64
);
define_float_ops!(f64x8, f32x16);
define_integer_ops!(
    (u64x8, u64),
    (i64x8, i64),
    (u32x16, u32),
    (i32x16, i32),
    (u16x32, u16),
    (i16x32, i16),
    (u8x64, u8),
    (i8x64, i8)
);
define_casts!(
    (f64x8, f32x8, as_f32x8),
    (f64x8, u64x8, as_u64x8),
    (f64x8, i64x8, as_i64x8),
    (f32x16, u32x16, as_u32x16),
    (f32x16, i32x16, as_i32x16),
    (u64x8, f64x8, as_f64x8),
    (u64x8, i64x8, as_i64x8),
    (i64x8, f64x8, as_f64x8),
    (i64x8, u64x8, as_u64x8),
    (u32x16, f32x16, as_f32x16),
    (u32x16, i32x16, as_i32x16),
    (i32x16, f32x16, as_f32x16),
    (i32x16, u32x16, as_u32x16),
    (u16x32, i16x32, as_i16x32),
    (i16x32, u16x32, as_u16x32),
    (u8x64, i8x64, as_i8x64),
    (i8x64, u8x64, as_u8x64)
);
