use crate::simd::{FixEndianness, Mask, MaskElement, Simd, SimdElement};

/// Choose elements from two vectors using a mask.
///
/// For each element in the mask, choose the corresponding element from `true_values` if
/// that element mask is true, and `false_values` if that element mask is false.
///
/// If the mask is `u64`, it's treated as a bitmask with the least significant bit
/// corresponding to the first element.
///
/// # Examples
///
/// ## Selecting values from `Simd`
/// ```
/// # #![feature(portable_simd)]
/// # #[cfg(feature = "as_crate")] use core_simd::simd;
/// # #[cfg(not(feature = "as_crate"))] use core::simd;
/// # use simd::{Simd, Mask, Select};
/// let a = Simd::from_array([0, 1, 2, 3]);
/// let b = Simd::from_array([4, 5, 6, 7]);
/// let mask = Mask::<i32, 4>::from_array([true, false, false, true]);
/// let c = mask.select(a, b);
/// assert_eq!(c.to_array(), [0, 5, 6, 3]);
/// ```
///
/// ## Selecting values from `Mask`
/// ```
/// # #![feature(portable_simd)]
/// # #[cfg(feature = "as_crate")] use core_simd::simd;
/// # #[cfg(not(feature = "as_crate"))] use core::simd;
/// # use simd::{Mask, Select};
/// let a = Mask::<i32, 4>::from_array([true, true, false, false]);
/// let b = Mask::<i32, 4>::from_array([false, false, true, true]);
/// let mask = Mask::<i32, 4>::from_array([true, false, false, true]);
/// let c = mask.select(a, b);
/// assert_eq!(c.to_array(), [true, false, true, false]);
/// ```
///
/// ## Selecting with a bitmask
/// ```
/// # #![feature(portable_simd)]
/// # #[cfg(feature = "as_crate")] use core_simd::simd;
/// # #[cfg(not(feature = "as_crate"))] use core::simd;
/// # use simd::{Mask, Select};
/// let a = Mask::<i32, 4>::from_array([true, true, false, false]);
/// let b = Mask::<i32, 4>::from_array([false, false, true, true]);
/// let mask = 0b1001;
/// let c = mask.select(a, b);
/// assert_eq!(c.to_array(), [true, false, true, false]);
/// ```
pub trait Select<T> {
    /// Choose elements
    fn select(self, true_values: T, false_values: T) -> T;
}

impl<T, U, const N: usize> Select<Simd<T, N>> for Mask<U, N>
where
    T: SimdElement,
    U: MaskElement,
{
    #[inline]
    fn select(self, true_values: Simd<T, N>, false_values: Simd<T, N>) -> Simd<T, N> {
        // Safety:
        // simd_as between masks is always safe (they're vectors of ints).
        // simd_select uses a mask that matches the width and number of elements
        unsafe {
            let mask: Simd<T::Mask, N> = core::intrinsics::simd::simd_as(self.to_simd());
            core::intrinsics::simd::simd_select(mask, true_values, false_values)
        }
    }
}

impl<T, const N: usize> Select<Simd<T, N>> for u64
where
    T: SimdElement,
{
    #[inline]
    fn select(self, true_values: Simd<T, N>, false_values: Simd<T, N>) -> Simd<T, N> {
        const {
            assert!(N <= 64, "number of elements can't be greater than 64");
        }

        #[inline]
        unsafe fn select_impl<T, U: FixEndianness, const M: usize, const N: usize>(
            bitmask: U,
            true_values: Simd<T, N>,
            false_values: Simd<T, N>,
        ) -> Simd<T, N>
        where
            T: SimdElement,
        {
            let default = true_values[0];
            let true_values = true_values.resize::<M>(default);
            let false_values = false_values.resize::<M>(default);

            // LLVM assumes bit order should match endianness
            let bitmask = bitmask.fix_endianness();

            // Safety: the caller guarantees that the size of U matches M
            let selected = unsafe {
                core::intrinsics::simd::simd_select_bitmask(bitmask, true_values, false_values)
            };

            selected.resize::<N>(default)
        }

        // TODO modify simd_bitmask_select to truncate input, making this unnecessary
        if N <= 8 {
            let bitmask = self as u8;
            // Safety: bitmask matches length
            unsafe { select_impl::<T, u8, 8, N>(bitmask, true_values, false_values) }
        } else if N <= 16 {
            let bitmask = self as u16;
            // Safety: bitmask matches length
            unsafe { select_impl::<T, u16, 16, N>(bitmask, true_values, false_values) }
        } else if N <= 32 {
            let bitmask = self as u32;
            // Safety: bitmask matches length
            unsafe { select_impl::<T, u32, 32, N>(bitmask, true_values, false_values) }
        } else {
            let bitmask = self;
            // Safety: bitmask matches length
            unsafe { select_impl::<T, u64, 64, N>(bitmask, true_values, false_values) }
        }
    }
}

impl<T, U, const N: usize> Select<Mask<T, N>> for Mask<U, N>
where
    T: MaskElement,
    U: MaskElement,
{
    #[inline]
    fn select(self, true_values: Mask<T, N>, false_values: Mask<T, N>) -> Mask<T, N> {
        let selected: Simd<T, N> =
            Select::select(self, true_values.to_simd(), false_values.to_simd());

        // Safety: all values come from masks
        unsafe { Mask::from_simd_unchecked(selected) }
    }
}

impl<T, const N: usize> Select<Mask<T, N>> for u64
where
    T: MaskElement,
{
    #[inline]
    fn select(self, true_values: Mask<T, N>, false_values: Mask<T, N>) -> Mask<T, N> {
        let selected: Simd<T, N> =
            Select::select(self, true_values.to_simd(), false_values.to_simd());

        // Safety: all values come from masks
        unsafe { Mask::from_simd_unchecked(selected) }
    }
}
