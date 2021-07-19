use crate::intrinsics;
use crate::masks::*;
use crate::vector::ptr::{SimdConstPtr, SimdMutPtr};
use crate::vector::*;

/// A representation of a vector as an "array" with indices, implementing
/// operations applicable to any vector type based solely on "having lanes",
/// and describing relationships between vector and scalar types.
pub trait SimdArray<const LANES: usize>: crate::LanesAtMost32
where
    SimdUsize<LANES>: crate::LanesAtMost32,
    SimdIsize<LANES>: crate::LanesAtMost32,
    MaskSize<LANES>: crate::Mask,
    Self: Sized,
{
    /// The scalar type in every lane of this vector type.
    type Scalar: Copy + Sized;
    /// The number of lanes for this vector.
    const LANES: usize = LANES;

    /// Generates a SIMD vector with the same value in every lane.
    #[must_use]
    fn splat(val: Self::Scalar) -> Self;

    /// SIMD gather: construct a SIMD vector by reading from a slice, using potentially discontiguous indices.
    /// If an index is out of bounds, that lane instead selects the value from the "or" vector.
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core_simd::*;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = SimdUsize::<4>::from_array([9, 3, 0, 5]);
    /// let alt = SimdI32::from_array([-5, -4, -3, -2]);
    ///
    /// let result = SimdI32::<4>::gather_or(&vec, idxs, alt); // Note the lane that is out-of-bounds.
    /// assert_eq!(result, SimdI32::from_array([-5, 13, 10, 15]));
    /// ```
    #[must_use]
    #[inline]
    fn gather_or(slice: &[Self::Scalar], idxs: SimdUsize<LANES>, or: Self) -> Self {
        Self::gather_select(slice, MaskSize::splat(true), idxs, or)
    }

    /// SIMD gather: construct a SIMD vector by reading from a slice, using potentially discontiguous indices.
    /// Out-of-bounds indices instead use the default value for that lane (0).
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core_simd::*;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = SimdUsize::<4>::from_array([9, 3, 0, 5]);
    ///
    /// let result = SimdI32::<4>::gather_or_default(&vec, idxs); // Note the lane that is out-of-bounds.
    /// assert_eq!(result, SimdI32::from_array([0, 13, 10, 15]));
    /// ```
    #[must_use]
    #[inline]
    fn gather_or_default(slice: &[Self::Scalar], idxs: SimdUsize<LANES>) -> Self
    where
        Self::Scalar: Default,
    {
        Self::gather_or(slice, idxs, Self::splat(Self::Scalar::default()))
    }

    /// SIMD gather: construct a SIMD vector by reading from a slice, using potentially discontiguous indices.
    /// Out-of-bounds or masked indices instead select the value from the "or" vector.
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core_simd::*;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = SimdUsize::<4>::from_array([9, 3, 0, 5]);
    /// let alt = SimdI32::from_array([-5, -4, -3, -2]);
    /// let mask = MaskSize::from_array([true, true, true, false]); // Note the mask of the last lane.
    ///
    /// let result = SimdI32::<4>::gather_select(&vec, mask, idxs, alt); // Note the lane that is out-of-bounds.
    /// assert_eq!(result, SimdI32::from_array([-5, 13, 10, -2]));
    /// ```
    #[must_use]
    #[inline]
    fn gather_select(
        slice: &[Self::Scalar],
        mask: MaskSize<LANES>,
        idxs: SimdUsize<LANES>,
        or: Self,
    ) -> Self {
        let mask = (mask & idxs.lanes_lt(SimdUsize::splat(slice.len()))).to_int();
        let base_ptr = SimdConstPtr::splat(slice.as_ptr());
        // Ferris forgive me, I have done pointer arithmetic here.
        let ptrs = base_ptr.wrapping_add(idxs);
        // SAFETY: The ptrs have been bounds-masked to prevent memory-unsafe reads insha'allah
        unsafe { intrinsics::simd_gather(or, ptrs, mask) }
    }

    /// SIMD scatter: write a SIMD vector's values into a slice, using potentially discontiguous indices.
    /// Out-of-bounds indices are not written.
    /// `scatter` writes "in order", so if an index receives two writes, only the last is guaranteed.
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core_simd::*;
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = SimdUsize::<4>::from_array([9, 3, 0, 0]);
    /// let vals = SimdI32::from_array([-27, 82, -41, 124]);
    ///
    /// vals.scatter(&mut vec, idxs); // index 0 receives two writes.
    /// assert_eq!(vec, vec![124, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    #[inline]
    fn scatter(self, slice: &mut [Self::Scalar], idxs: SimdUsize<LANES>) {
        self.scatter_select(slice, MaskSize::splat(true), idxs)
    }

    /// SIMD scatter: write a SIMD vector's values into a slice, using potentially discontiguous indices.
    /// Out-of-bounds or masked indices are not written.
    /// `scatter_select` writes "in order", so if an index receives two writes, only the last is guaranteed.
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core_simd::*;
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = SimdUsize::<4>::from_array([9, 3, 0, 0]);
    /// let vals = SimdI32::from_array([-27, 82, -41, 124]);
    /// let mask = MaskSize::from_array([true, true, true, false]); // Note the mask of the last lane.
    ///
    /// vals.scatter_select(&mut vec, mask, idxs); // index 0's second write is masked, thus omitted.
    /// assert_eq!(vec, vec![-41, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    #[inline]
    fn scatter_select(
        self,
        slice: &mut [Self::Scalar],
        mask: MaskSize<LANES>,
        idxs: SimdUsize<LANES>,
    ) {
        // We must construct our scatter mask before we derive a pointer!
        let mask = (mask & idxs.lanes_lt(SimdUsize::splat(slice.len()))).to_int();
        // SAFETY: This block works with *mut T derived from &mut 'a [T],
        // which means it is delicate in Rust's borrowing model, circa 2021:
        // &mut 'a [T] asserts uniqueness, so deriving &'a [T] invalidates live *mut Ts!
        // Even though this block is largely safe methods, it must be almost exactly this way
        // to prevent invalidating the raw ptrs while they're live.
        // Thus, entering this block requires all values to use being already ready:
        // 0. idxs we want to write to, which are used to construct the mask.
        // 1. mask, which depends on an initial &'a [T] and the idxs.
        // 2. actual values to scatter (self).
        // 3. &mut [T] which will become our base ptr.
        unsafe {
            // Now Entering ☢️ *mut T Zone
            let base_ptr = SimdMutPtr::splat(slice.as_mut_ptr());
            // Ferris forgive me, I have done pointer arithmetic here.
            let ptrs = base_ptr.wrapping_add(idxs);
            // The ptrs have been bounds-masked to prevent memory-unsafe writes insha'allah
            intrinsics::simd_scatter(self, ptrs, mask)
            // Cleared ☢️ *mut T Zone
        }
    }
}

macro_rules! impl_simdarray_for {
    ($simd:ident {type Scalar = $scalar:ident;}) => {
        impl<const LANES: usize> SimdArray<LANES> for $simd<LANES>
            where SimdUsize<LANES>: crate::LanesAtMost32,
            SimdIsize<LANES>: crate::LanesAtMost32,
            MaskSize<LANES>: crate::Mask,
            Self: crate::LanesAtMost32,
        {
            type Scalar = $scalar;

            #[must_use]
            #[inline]
            fn splat(val: Self::Scalar) -> Self {
                [val; LANES].into()
            }
        }
    };

    ($simd:ident $impl:tt) => {
        impl<const LANES: usize> SimdArray<LANES> for $simd<LANES>
            where SimdUsize<LANES>: crate::LanesAtMost32,
            SimdIsize<LANES>: crate::LanesAtMost32,
            MaskSize<LANES>: crate::Mask,
            Self: crate::LanesAtMost32,
        $impl
    }
}

impl_simdarray_for! {
    SimdUsize {
        type Scalar = usize;
    }
}

impl_simdarray_for! {
    SimdIsize {
        type Scalar = isize;
    }
}

impl_simdarray_for! {
    SimdI8 {
        type Scalar = i8;
    }
}

impl_simdarray_for! {
    SimdI16 {
        type Scalar = i16;
    }
}

impl_simdarray_for! {
    SimdI32 {
        type Scalar = i32;
    }
}

impl_simdarray_for! {
    SimdI64 {
        type Scalar = i64;
    }
}

impl_simdarray_for! {
    SimdU8 {
        type Scalar = u8;
    }
}

impl_simdarray_for! {
    SimdU16 {
        type Scalar = u16;
    }
}

impl_simdarray_for! {
    SimdU32 {
        type Scalar = u32;
    }
}

impl_simdarray_for! {
    SimdU64 {
        type Scalar = u64;
    }
}

impl_simdarray_for! {
    SimdF32 {
        type Scalar = f32;
    }
}

impl_simdarray_for! {
    SimdF64 {
        type Scalar = f64;
    }
}
