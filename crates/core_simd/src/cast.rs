use crate::simd::SimdElement;

/// Supporting trait for `Simd::cast`.  Typically doesn't need to be used directly.
pub unsafe trait SimdCast: SimdElement {}

unsafe impl SimdCast for i8 {}
unsafe impl SimdCast for i16 {}
unsafe impl SimdCast for i32 {}
unsafe impl SimdCast for i64 {}
unsafe impl SimdCast for isize {}
unsafe impl SimdCast for u8 {}
unsafe impl SimdCast for u16 {}
unsafe impl SimdCast for u32 {}
unsafe impl SimdCast for u64 {}
unsafe impl SimdCast for usize {}
unsafe impl SimdCast for f32 {}
unsafe impl SimdCast for f64 {}

/// Supporting trait for `Simd::cast_ptr`.  Typically doesn't need to be used directly.
pub unsafe trait SimdCastPtr: SimdElement {}

unsafe impl<T> SimdCastPtr for *const T {}
unsafe impl<T> SimdCastPtr for *mut T {}
