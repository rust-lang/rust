#![feature(repr_simd)]

#[macro_use]
mod macros;

macro_rules! import_types {
    { $($mod:ident,)* } => {
        $(
        mod $mod;
        pub use $mod::*;
        )*
    }
}

import_types! {
    type_u8x2,   type_u8x4,   type_u8x8,   type_u8x16,   type_u8x32,   type_u8x64,
    type_i8x2,   type_i8x4,   type_i8x8,   type_i8x16,   type_i8x32,   type_i8x64,
    type_u16x2,  type_u16x4,  type_u16x8,  type_u16x16,  type_u16x32,
    type_i16x2,  type_i16x4,  type_i16x8,  type_i16x16,  type_i16x32,
    type_u32x2,  type_u32x4,  type_u32x8,  type_u32x16,
    type_i32x2,  type_i32x4,  type_i32x8,  type_i32x16,
    type_u64x2,  type_u64x4,  type_u64x8,
    type_i64x2,  type_i64x4,  type_i64x8,
    type_u128x2, type_u128x4,
    type_i128x2, type_i128x4,
}

import_types! {
    type_usizex2, type_usizex4, type_usizex8,
    type_isizex2, type_isizex4, type_isizex8,
}

import_types! {
    type_f32x2, type_f32x4, type_f32x8, type_f32x16,
    type_f64x2, type_f64x4, type_f64x8,
}
