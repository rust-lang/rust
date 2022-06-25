use crate::simd::{intrinsics, LaneCount, Simd, SimdElement, SupportedLaneCount};

/// Supporting trait for `Simd::cast`.  Typically doesn't need to be used directly.
pub unsafe trait SimdCast<Target: SimdElement>: SimdElement {
    #[doc(hidden)]
    fn cast<const LANES: usize>(x: Simd<Self, LANES>) -> Simd<Target, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Safety: implementing this trait indicates that the types are supported by `simd_as`
        unsafe { intrinsics::simd_as(x) }
    }

    #[doc(hidden)]
    unsafe fn cast_unchecked<const LANES: usize>(x: Simd<Self, LANES>) -> Simd<Target, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Safety: implementing this trait indicates that the types are supported by `simd_cast`
        // The caller is responsible for the conversion invariants.
        unsafe { intrinsics::simd_cast(x) }
    }
}

macro_rules! into_number {
    { unsafe $from:ty as $to:ty } => {
        // Safety: casting between numbers is supported by `simd_cast` and `simd_as`
        unsafe impl SimdCast<$to> for $from {}
    };
    { unsafe $($type:ty),* } => {
        $(
        into_number! { unsafe $type as i8 }
        into_number! { unsafe $type as i16 }
        into_number! { unsafe $type as i32 }
        into_number! { unsafe $type as i64 }
        into_number! { unsafe $type as isize }

        into_number! { unsafe $type as u8 }
        into_number! { unsafe $type as u16 }
        into_number! { unsafe $type as u32 }
        into_number! { unsafe $type as u64 }
        into_number! { unsafe $type as usize }

        into_number! { unsafe $type as f32 }
        into_number! { unsafe $type as f64 }
        )*
    }
}

into_number! { unsafe i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64 }

// TODO uncomment pending PR to rustc
/*
macro_rules! into_pointer {
    { unsafe $($type:ty),* } => {
        $(
        // Safety: casting between numbers and pointers is supported by `simd_cast` and `simd_as`
        unsafe impl<T> SimdCast<$type> for *const T {}
        // Safety: casting between numbers and pointers is supported by `simd_cast` and `simd_as`
        unsafe impl<T> SimdCast<$type> for *mut T {}
        // Safety: casting between numbers and pointers is supported by `simd_cast` and `simd_as`
        unsafe impl<T> SimdCast<*const T> for $type {}
        // Safety: casting between numbers and pointers is supported by `simd_cast` and `simd_as`
        unsafe impl<T> SimdCast<*mut T> for $type {}
        )*
    }
}

into_pointer! { unsafe i8, i16, i32, i64, isize, u8, u16, u32, u64, usize }

// Safety: casting between pointers is supported by `simd_cast` and `simd_as`
unsafe impl<T, U> SimdCast<*const T> for *const U {}
// Safety: casting between pointers is supported by `simd_cast` and `simd_as`
unsafe impl<T, U> SimdCast<*const T> for *mut U {}
// Safety: casting between pointers is supported by `simd_cast` and `simd_as`
unsafe impl<T, U> SimdCast<*mut T> for *const U {}
// Safety: casting between pointers is supported by `simd_cast` and `simd_as`
unsafe impl<T, U> SimdCast<*mut T> for *mut U {}
*/
