#![allow(non_camel_case_types)]

use crate::simd::Simd;

/// A SIMD vector with two elements of type `isize`.
pub type isizex2 = Simd<isize, 2>;

/// A SIMD vector with four elements of type `isize`.
pub type isizex4 = Simd<isize, 4>;

/// A SIMD vector with eight elements of type `isize`.
pub type isizex8 = Simd<isize, 8>;

/// A 32-bit SIMD vector with two elements of type `i16`.
pub type i16x2 = Simd<i16, 2>;

/// A 64-bit SIMD vector with four elements of type `i16`.
pub type i16x4 = Simd<i16, 4>;

/// A 128-bit SIMD vector with eight elements of type `i16`.
pub type i16x8 = Simd<i16, 8>;

/// A 256-bit SIMD vector with 16 elements of type `i16`.
pub type i16x16 = Simd<i16, 16>;

/// A 512-bit SIMD vector with 32 elements of type `i16`.
pub type i16x32 = Simd<i16, 32>;

/// A 64-bit SIMD vector with two elements of type `i32`.
pub type i32x2 = Simd<i32, 2>;

/// A 128-bit SIMD vector with four elements of type `i32`.
pub type i32x4 = Simd<i32, 4>;

/// A 256-bit SIMD vector with eight elements of type `i32`.
pub type i32x8 = Simd<i32, 8>;

/// A 512-bit SIMD vector with 16 elements of type `i32`.
pub type i32x16 = Simd<i32, 16>;

/// A 128-bit SIMD vector with two elements of type `i64`.
pub type i64x2 = Simd<i64, 2>;

/// A 256-bit SIMD vector with four elements of type `i64`.
pub type i64x4 = Simd<i64, 4>;

/// A 512-bit SIMD vector with eight elements of type `i64`.
pub type i64x8 = Simd<i64, 8>;

/// A 32-bit SIMD vector with four elements of type `i8`.
pub type i8x4 = Simd<i8, 4>;

/// A 64-bit SIMD vector with eight elements of type `i8`.
pub type i8x8 = Simd<i8, 8>;

/// A 128-bit SIMD vector with 16 elements of type `i8`.
pub type i8x16 = Simd<i8, 16>;

/// A 256-bit SIMD vector with 32 elements of type `i8`.
pub type i8x32 = Simd<i8, 32>;

/// A 512-bit SIMD vector with 64 elements of type `i8`.
pub type i8x64 = Simd<i8, 64>;
