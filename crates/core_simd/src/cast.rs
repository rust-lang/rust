use crate::simd::{intrinsics, LaneCount, Simd, SimdElement, SupportedLaneCount};

/// Supporting trait for `Simd::cast`.  Typically doesn't need to be used directly.
pub trait SimdCast<Target: SimdElement>: SimdElement {
    #[doc(hidden)]
    fn cast<const LANES: usize>(x: Simd<Self, LANES>) -> Simd<Target, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

macro_rules! into_number {
    { $from:ty, $to:ty } => {
        impl SimdCast<$to> for $from {
            fn cast<const LANES: usize>(x: Simd<Self, LANES>) -> Simd<$to, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                // Safety: simd_as can handle numeric conversions
                unsafe { intrinsics::simd_as(x) }
            }
        }
    };
    { $($type:ty),* } => {
        $(
        into_number! { $type, i8 }
        into_number! { $type, i16 }
        into_number! { $type, i32 }
        into_number! { $type, i64 }
        into_number! { $type, isize }

        into_number! { $type, u8 }
        into_number! { $type, u16 }
        into_number! { $type, u32 }
        into_number! { $type, u64 }
        into_number! { $type, usize }

        into_number! { $type, f32 }
        into_number! { $type, f64 }
        )*
    }
}

into_number! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64 }

macro_rules! into_pointer {
    { $($type:ty),* } => {
        $(
        impl<T> SimdCast<$type> for *const T {
            fn cast<const LANES: usize>(x: Simd<Self, LANES>) -> Simd<$type, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                // Safety: transmuting isize to pointers is safe
                let x: Simd<isize, LANES> = unsafe { core::mem::transmute_copy(&x) };
                x.cast()
            }
        }
        impl<T> SimdCast<$type> for *mut T {
            fn cast<const LANES: usize>(x: Simd<Self, LANES>) -> Simd<$type, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                // Safety: transmuting isize to pointers is safe
                let x: Simd<isize, LANES> = unsafe { core::mem::transmute_copy(&x) };
                x.cast()
            }
        }
        impl<T> SimdCast<*const T> for $type {
            fn cast<const LANES: usize>(x: Simd<$type, LANES>) -> Simd<*const T, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                let x: Simd<isize, LANES> = x.cast();
                // Safety: transmuting isize to pointers is safe
                unsafe { core::mem::transmute_copy(&x) }
            }
        }
        impl<T> SimdCast<*mut T> for $type {
            fn cast<const LANES: usize>(x: Simd<$type, LANES>) -> Simd<*mut T, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                let x: Simd<isize, LANES> = x.cast();
                // Safety: transmuting isize to pointers is safe
                unsafe { core::mem::transmute_copy(&x) }
            }
        }
        )*
    }
}

into_pointer! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize }

impl<T, U> SimdCast<*const T> for *const U {
    fn cast<const LANES: usize>(x: Simd<*const U, LANES>) -> Simd<*const T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Safety: transmuting pointers is safe
        unsafe { core::mem::transmute_copy(&x) }
    }
}
impl<T, U> SimdCast<*const T> for *mut U {
    fn cast<const LANES: usize>(x: Simd<*mut U, LANES>) -> Simd<*const T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Safety: transmuting pointers is safe
        unsafe { core::mem::transmute_copy(&x) }
    }
}
impl<T, U> SimdCast<*mut T> for *const U {
    fn cast<const LANES: usize>(x: Simd<*const U, LANES>) -> Simd<*mut T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Safety: transmuting pointers is safe
        unsafe { core::mem::transmute_copy(&x) }
    }
}
impl<T, U> SimdCast<*mut T> for *mut U {
    fn cast<const LANES: usize>(x: Simd<*mut U, LANES>) -> Simd<*mut T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Safety: transmuting pointers is safe
        unsafe { core::mem::transmute_copy(&x) }
    }
}
