//! The portable SIMD prelude.
//!
//! Includes important traits and types to be imported with a glob:
//! ```ignore
//! use std::simd::prelude::*;
//! ```

#[doc(no_inline)]
pub use super::{
    simd_swizzle, Mask, Simd, SimdConstPtr, SimdFloat, SimdInt, SimdMutPtr, SimdOrd, SimdPartialEq,
    SimdPartialOrd, SimdUint,
};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{f32x1, f32x2, f32x4, f32x8, f32x16, f32x32, f32x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{f64x1, f64x2, f64x4, f64x8, f64x16, f64x32, f64x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{i8x1, i8x2, i8x4, i8x8, i8x16, i8x32, i8x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{i16x1, i16x2, i16x4, i16x8, i16x16, i16x32, i16x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{i32x1, i32x2, i32x4, i32x8, i32x16, i32x32, i32x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{i64x1, i64x2, i64x4, i64x8, i64x16, i64x32, i64x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{isizex1, isizex2, isizex4, isizex8, isizex16, isizex32, isizex64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{u8x1, u8x2, u8x4, u8x8, u8x16, u8x32, u8x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{u16x1, u16x2, u16x4, u16x8, u16x16, u16x32, u16x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{u32x1, u32x2, u32x4, u32x8, u32x16, u32x32, u32x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{u64x1, u64x2, u64x4, u64x8, u64x16, u64x32, u64x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{usizex1, usizex2, usizex4, usizex8, usizex16, usizex32, usizex64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{mask8x1, mask8x2, mask8x4, mask8x8, mask8x16, mask8x32, mask8x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{mask16x1, mask16x2, mask16x4, mask16x8, mask16x16, mask16x32, mask16x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{mask32x1, mask32x2, mask32x4, mask32x8, mask32x16, mask32x32, mask32x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{mask64x1, mask64x2, mask64x4, mask64x8, mask64x16, mask64x32, mask64x64};

#[rustfmt::skip]
#[doc(no_inline)]
pub use super::{masksizex1, masksizex2, masksizex4, masksizex8, masksizex16, masksizex32, masksizex64};
