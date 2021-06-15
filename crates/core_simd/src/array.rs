use crate::intrinsics;
use crate::masks::*;
use crate::vector::ptr::SimdConstPtr;
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

    /// Generates a SIMD vector with the same value in every lane.
    #[must_use]
    fn splat(val: Self::Scalar) -> Self;

    /// SIMD gather: construct a SIMD vector by reading from a slice, using potentially discontiguous indices.
    /// If an index is out of bounds, that lane instead selects the value from the "or" vector.
    /// ```
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
