use crate::simd::SimdElement;

mod sealed {
    /// Cast vector elements to other types.
    ///
    /// # Safety
    /// Implementing this trait asserts that the type is a valid vector element for the `simd_cast`
    /// or `simd_as` intrinsics.
    pub unsafe trait Sealed {}
}
use sealed::Sealed;

/// Supporting trait for `Simd::cast`.  Typically doesn't need to be used directly.
pub trait SimdCast: Sealed + SimdElement {}

// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for i8 {}
impl SimdCast for i8 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for i16 {}
impl SimdCast for i16 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for i32 {}
impl SimdCast for i32 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for i64 {}
impl SimdCast for i64 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for isize {}
impl SimdCast for isize {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for u8 {}
impl SimdCast for u8 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for u16 {}
impl SimdCast for u16 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for u32 {}
impl SimdCast for u32 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for u64 {}
impl SimdCast for u64 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for usize {}
impl SimdCast for usize {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for f32 {}
impl SimdCast for f32 {}
// Safety: primitive number types can be cast to other primitive number types
unsafe impl Sealed for f64 {}
impl SimdCast for f64 {}
