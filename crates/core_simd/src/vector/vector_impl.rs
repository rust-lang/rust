/// Implements common traits on the specified vector `$name`, holding multiple `$lanes` of `$type`.
macro_rules! impl_vector {
    { $name:ident, $type:ty } => {
        impl<const LANES: usize> crate::vector::sealed::Sealed for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {}

        impl<const LANES: usize> crate::vector::Vector for $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            type Scalar = $type;
            const LANES: usize = LANES;

            #[inline]
            fn splat(val: Self::Scalar) -> Self {
                Self::splat(val)
            }
        }

        impl<const LANES: usize> $name<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            /// Construct a SIMD vector by setting all lanes to the given value.
            pub const fn splat(value: $type) -> Self {
                Self([value; LANES])
            }

            /// Returns an array reference containing the entire SIMD vector.
            pub const fn as_array(&self) -> &[$type; LANES] {
                &self.0
            }

            /// Returns a mutable array reference containing the entire SIMD vector.
            pub fn as_mut_array(&mut self) -> &mut [$type; LANES] {
                &mut self.0
            }

            /// Converts an array to a SIMD vector.
            pub const fn from_array(array: [$type; LANES]) -> Self {
                Self(array)
            }

            /// Converts a SIMD vector to an array.
            pub const fn to_array(self) -> [$type; LANES] {
                self.0
            }

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
            pub fn gather_or(slice: &[$type], idxs: crate::SimdUsize<LANES>, or: Self) -> Self {
                Self::gather_select(slice, crate::MaskSize::splat(true), idxs, or)
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
            pub fn gather_or_default(slice: &[$type], idxs: crate::SimdUsize<LANES>) -> Self {
                Self::gather_or(slice, idxs, Self::splat(<$type>::default()))
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
            pub fn gather_select(
                slice: &[$type],
                mask: crate::MaskSize<LANES>,
                idxs: crate::SimdUsize<LANES>,
                or: Self,
            ) -> Self
            {
                let mask = (mask & idxs.lanes_lt(crate::SimdUsize::splat(slice.len()))).to_int();
                let base_ptr = crate::vector::ptr::SimdConstPtr::splat(slice.as_ptr());
                // Ferris forgive me, I have done pointer arithmetic here.
                let ptrs = base_ptr.wrapping_add(idxs);
                // SAFETY: The ptrs have been bounds-masked to prevent memory-unsafe reads insha'allah
                unsafe { crate::intrinsics::simd_gather(or, ptrs, mask) }
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
            pub fn scatter(self, slice: &mut [$type], idxs: crate::SimdUsize<LANES>) {
                self.scatter_select(slice, crate::MaskSize::splat(true), idxs)
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
            pub fn scatter_select(
                self,
                slice: &mut [$type],
                mask: crate::MaskSize<LANES>,
                idxs: crate::SimdUsize<LANES>,
            )
            {
                // We must construct our scatter mask before we derive a pointer!
                let mask = (mask & idxs.lanes_lt(crate::SimdUsize::splat(slice.len()))).to_int();
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
                    let base_ptr = crate::vector::ptr::SimdMutPtr::splat(slice.as_mut_ptr());
                    // Ferris forgive me, I have done pointer arithmetic here.
                    let ptrs = base_ptr.wrapping_add(idxs);
                    // The ptrs have been bounds-masked to prevent memory-unsafe writes insha'allah
                    crate::intrinsics::simd_scatter(self, ptrs, mask)
                    // Cleared ☢️ *mut T Zone
                }
            }
        }

        impl<const LANES: usize> Copy for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {}

        impl<const LANES: usize> Clone for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const LANES: usize> Default for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn default() -> Self {
                Self::splat(<$type>::default())
            }
        }

        impl<const LANES: usize> PartialEq for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                // TODO use SIMD equality
                self.to_array() == other.to_array()
            }
        }

        impl<const LANES: usize> PartialOrd for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                // TODO use SIMD equalitya
                self.to_array().partial_cmp(other.as_ref())
            }
        }

        // array references
        impl<const LANES: usize> AsRef<[$type; LANES]> for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn as_ref(&self) -> &[$type; LANES] {
                &self.0
            }
        }

        impl<const LANES: usize> AsMut<[$type; LANES]> for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn as_mut(&mut self) -> &mut [$type; LANES] {
                &mut self.0
            }
        }

        // slice references
        impl<const LANES: usize> AsRef<[$type]> for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn as_ref(&self) -> &[$type] {
                &self.0
            }
        }

        impl<const LANES: usize> AsMut<[$type]> for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            #[inline]
            fn as_mut(&mut self) -> &mut [$type] {
                &mut self.0
            }
        }

        // vector/array conversion
        impl<const LANES: usize> From<[$type; LANES]> for $name<LANES> where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            fn from(array: [$type; LANES]) -> Self {
                Self(array)
            }
        }

        impl <const LANES: usize> From<$name<LANES>> for [$type; LANES] where crate::LaneCount<LANES>: crate::SupportedLaneCount {
            fn from(vector: $name<LANES>) -> Self {
                vector.to_array()
            }
        }

        impl_shuffle_2pow_lanes!{ $name }
    }
}
