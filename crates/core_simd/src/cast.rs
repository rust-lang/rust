use crate::simd::SimdElement;

/// Supporting trait for `Simd::cast`.  Typically doesn't need to be used directly.
pub trait SimdCast<Target: SimdElement>: SimdElement {}

macro_rules! into_number {
    { $($type:ty),* } => {
        $(
        impl SimdCast<i8> for $type {}
        impl SimdCast<i16> for $type {}
        impl SimdCast<i32> for $type {}
        impl SimdCast<i64> for $type {}
        impl SimdCast<isize> for $type {}

        impl SimdCast<u8> for $type {}
        impl SimdCast<u16> for $type {}
        impl SimdCast<u32> for $type {}
        impl SimdCast<u64> for $type {}
        impl SimdCast<usize> for $type {}

        impl SimdCast<f32> for $type {}
        impl SimdCast<f64> for $type {}
        )*
    }
}

into_number! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64 }

macro_rules! into_pointer {
    { $($type:ty),* } => {
        $(
        impl<T> SimdCast<$type> for *const T {}
        impl<T> SimdCast<$type> for *mut T {}
        impl<T> SimdCast<*const T> for $type {}
        impl<T> SimdCast<*mut T> for $type {}
        )*
    }
}

into_pointer! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize }

impl<T, U> SimdCast<*const T> for *const U {}
impl<T, U> SimdCast<*const T> for *mut U {}
impl<T, U> SimdCast<*mut T> for *const U {}
impl<T, U> SimdCast<*mut T> for *mut U {}
