#[macro_use]
mod vector_impl;

mod float;
mod int;
mod uint;

pub use float::*;
pub use int::*;
pub use uint::*;

// Vectors of pointers are not for public use at the current time.
pub(crate) mod ptr;

use crate::{LaneCount, SupportedLaneCount};

/// A SIMD vector of `LANES` elements of type `Element`.
#[repr(simd)]
pub struct Simd<Element, const LANES: usize>([Element; LANES])
where
    Element: SimdElement,
    LaneCount<LANES>: SupportedLaneCount;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Marker trait for types that may be used as SIMD vector elements.
pub unsafe trait SimdElement: Sealed {
    /// The mask element type corresponding to this element type.
    type Mask: SimdElement;
}

impl Sealed for u8 {}
unsafe impl SimdElement for u8 {
    type Mask = u8;
}

impl Sealed for u16 {}
unsafe impl SimdElement for u16 {
    type Mask = u16;
}

impl Sealed for u32 {}
unsafe impl SimdElement for u32 {
    type Mask = u32;
}

impl Sealed for u64 {}
unsafe impl SimdElement for u64 {
    type Mask = u64;
}

impl Sealed for usize {}
unsafe impl SimdElement for usize {
    type Mask = usize;
}

impl Sealed for i8 {}
unsafe impl SimdElement for i8 {
    type Mask = i8;
}

impl Sealed for i16 {}
unsafe impl SimdElement for i16 {
    type Mask = i16;
}

impl Sealed for i32 {}
unsafe impl SimdElement for i32 {
    type Mask = i32;
}

impl Sealed for i64 {}
unsafe impl SimdElement for i64 {
    type Mask = i64;
}

impl Sealed for isize {}
unsafe impl SimdElement for isize {
    type Mask = isize;
}

impl Sealed for f32 {}
unsafe impl SimdElement for f32 {
    type Mask = i32;
}

impl Sealed for f64 {}
unsafe impl SimdElement for f64 {
    type Mask = i64;
}

/// A representation of a vector as an "array" with indices, implementing
/// operations applicable to any vector type based solely on "having lanes",
/// and describing relationships between vector and scalar types.
pub trait Vector: sealed::Sealed {
    /// The scalar type in every lane of this vector type.
    type Scalar: Copy + Sized;

    /// The number of lanes for this vector.
    const LANES: usize;

    /// Generates a SIMD vector with the same value in every lane.
    #[must_use]
    fn splat(val: Self::Scalar) -> Self;
}
