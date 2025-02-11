//! Types and traits associated with masking elements of vectors.
//! Types representing
#![allow(non_camel_case_types)]

#[cfg_attr(
    not(all(target_arch = "x86_64", target_feature = "avx512f")),
    path = "masks/full_masks.rs"
)]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    path = "masks/bitmask.rs"
)]
mod mask_impl;

use crate::simd::{LaneCount, Simd, SimdCast, SimdElement, SupportedLaneCount};
use core::cmp::Ordering;
use core::{fmt, mem};

mod sealed {
    use super::*;

    /// Not only does this seal the `MaskElement` trait, but these functions prevent other traits
    /// from bleeding into the parent bounds.
    ///
    /// For example, `eq` could be provided by requiring `MaskElement: PartialEq`, but that would
    /// prevent us from ever removing that bound, or from implementing `MaskElement` on
    /// non-`PartialEq` types in the future.
    pub trait Sealed {
        fn valid<const N: usize>(values: Simd<Self, N>) -> bool
        where
            LaneCount<N>: SupportedLaneCount,
            Self: SimdElement;

        fn eq(self, other: Self) -> bool;

        fn to_usize(self) -> usize;
        fn max_unsigned() -> u64;

        type Unsigned: SimdElement;

        const TRUE: Self;

        const FALSE: Self;
    }
}
use sealed::Sealed;

/// Marker trait for types that may be used as SIMD mask elements.
///
/// # Safety
/// Type must be a signed integer.
pub unsafe trait MaskElement: SimdElement<Mask = Self> + SimdCast + Sealed {}

macro_rules! impl_element {
    { $ty:ty, $unsigned:ty } => {
        impl Sealed for $ty {
            #[inline]
            fn valid<const N: usize>(value: Simd<Self, N>) -> bool
            where
                LaneCount<N>: SupportedLaneCount,
            {
                // We can't use `Simd` directly, because `Simd`'s functions call this function and
                // we will end up with an infinite loop.
                // Safety: `value` is an integer vector
                unsafe {
                    use core::intrinsics::simd;
                    let falses: Simd<Self, N> = simd::simd_eq(value, Simd::splat(0 as _));
                    let trues: Simd<Self, N> = simd::simd_eq(value, Simd::splat(-1 as _));
                    let valid: Simd<Self, N> = simd::simd_or(falses, trues);
                    simd::simd_reduce_all(valid)
                }
            }

            #[inline]
            fn eq(self, other: Self) -> bool { self == other }

            #[inline]
            fn to_usize(self) -> usize {
                self as usize
            }

            #[inline]
            fn max_unsigned() -> u64 {
                <$unsigned>::MAX as u64
            }

            type Unsigned = $unsigned;

            const TRUE: Self = -1;
            const FALSE: Self = 0;
        }

        // Safety: this is a valid mask element type
        unsafe impl MaskElement for $ty {}
    }
}

impl_element! { i8, u8 }
impl_element! { i16, u16 }
impl_element! { i32, u32 }
impl_element! { i64, u64 }
impl_element! { isize, usize }

/// A SIMD vector mask for `N` elements of width specified by `Element`.
///
/// Masks represent boolean inclusion/exclusion on a per-element basis.
///
/// The layout of this type is unspecified, and may change between platforms
/// and/or Rust versions, and code should not assume that it is equivalent to
/// `[T; N]`.
#[repr(transparent)]
pub struct Mask<T, const N: usize>(mask_impl::Mask<T, N>)
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount;

impl<T, const N: usize> Copy for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
}

impl<T, const N: usize> Clone for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    /// Constructs a mask by setting all elements to the given value.
    #[inline]
    pub fn splat(value: bool) -> Self {
        Self(mask_impl::Mask::splat(value))
    }

    /// Converts an array of bools to a SIMD mask.
    #[inline]
    pub fn from_array(array: [bool; N]) -> Self {
        // SAFETY: Rust's bool has a layout of 1 byte (u8) with a value of
        //     true:    0b_0000_0001
        //     false:   0b_0000_0000
        // Thus, an array of bools is also a valid array of bytes: [u8; N]
        // This would be hypothetically valid as an "in-place" transmute,
        // but these are "dependently-sized" types, so copy elision it is!
        unsafe {
            let bytes: [u8; N] = mem::transmute_copy(&array);
            let bools: Simd<i8, N> =
                core::intrinsics::simd::simd_ne(Simd::from_array(bytes), Simd::splat(0u8));
            Mask::from_int_unchecked(core::intrinsics::simd::simd_cast(bools))
        }
    }

    /// Converts a SIMD mask to an array of bools.
    #[inline]
    pub fn to_array(self) -> [bool; N] {
        // This follows mostly the same logic as from_array.
        // SAFETY: Rust's bool has a layout of 1 byte (u8) with a value of
        //     true:    0b_0000_0001
        //     false:   0b_0000_0000
        // Thus, an array of bools is also a valid array of bytes: [u8; N]
        // Since our masks are equal to integers where all bits are set,
        // we can simply convert them to i8s, and then bitand them by the
        // bitpattern for Rust's "true" bool.
        // This would be hypothetically valid as an "in-place" transmute,
        // but these are "dependently-sized" types, so copy elision it is!
        unsafe {
            let mut bytes: Simd<i8, N> = core::intrinsics::simd::simd_cast(self.to_int());
            bytes &= Simd::splat(1i8);
            mem::transmute_copy(&bytes)
        }
    }

    /// Converts a vector of integers to a mask, where 0 represents `false` and -1
    /// represents `true`.
    ///
    /// # Safety
    /// All elements must be either 0 or -1.
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub unsafe fn from_int_unchecked(value: Simd<T, N>) -> Self {
        // Safety: the caller must confirm this invariant
        unsafe {
            core::intrinsics::assume(<T as Sealed>::valid(value));
            Self(mask_impl::Mask::from_int_unchecked(value))
        }
    }

    /// Converts a vector of integers to a mask, where 0 represents `false` and -1
    /// represents `true`.
    ///
    /// # Panics
    /// Panics if any element is not 0 or -1.
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    #[track_caller]
    pub fn from_int(value: Simd<T, N>) -> Self {
        assert!(T::valid(value), "all values must be either 0 or -1",);
        // Safety: the validity has been checked
        unsafe { Self::from_int_unchecked(value) }
    }

    /// Converts the mask to a vector of integers, where 0 represents `false` and -1
    /// represents `true`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn to_int(self) -> Simd<T, N> {
        self.0.to_int()
    }

    /// Converts the mask to a mask of any other element size.
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn cast<U: MaskElement>(self) -> Mask<U, N> {
        Mask(self.0.convert())
    }

    /// Tests the value of the specified element.
    ///
    /// # Safety
    /// `index` must be less than `self.len()`.
    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub unsafe fn test_unchecked(&self, index: usize) -> bool {
        // Safety: the caller must confirm this invariant
        unsafe { self.0.test_unchecked(index) }
    }

    /// Tests the value of the specified element.
    ///
    /// # Panics
    /// Panics if `index` is greater than or equal to the number of elements in the vector.
    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    #[track_caller]
    pub fn test(&self, index: usize) -> bool {
        assert!(index < N, "element index out of range");
        // Safety: the element index has been checked
        unsafe { self.test_unchecked(index) }
    }

    /// Sets the value of the specified element.
    ///
    /// # Safety
    /// `index` must be less than `self.len()`.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        // Safety: the caller must confirm this invariant
        unsafe {
            self.0.set_unchecked(index, value);
        }
    }

    /// Sets the value of the specified element.
    ///
    /// # Panics
    /// Panics if `index` is greater than or equal to the number of elements in the vector.
    #[inline]
    #[track_caller]
    pub fn set(&mut self, index: usize, value: bool) {
        assert!(index < N, "element index out of range");
        // Safety: the element index has been checked
        unsafe {
            self.set_unchecked(index, value);
        }
    }

    /// Returns true if any element is set, or false otherwise.
    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub fn any(self) -> bool {
        self.0.any()
    }

    /// Returns true if all elements are set, or false otherwise.
    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    pub fn all(self) -> bool {
        self.0.all()
    }

    /// Creates a bitmask from a mask.
    ///
    /// Each bit is set if the corresponding element in the mask is `true`.
    /// If the mask contains more than 64 elements, the bitmask is truncated to the first 64.
    #[inline]
    #[must_use = "method returns a new integer and does not mutate the original value"]
    pub fn to_bitmask(self) -> u64 {
        self.0.to_bitmask_integer()
    }

    /// Creates a mask from a bitmask.
    ///
    /// For each bit, if it is set, the corresponding element in the mask is set to `true`.
    /// If the mask contains more than 64 elements, the remainder are set to `false`.
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    pub fn from_bitmask(bitmask: u64) -> Self {
        Self(mask_impl::Mask::from_bitmask_integer(bitmask))
    }

    /// Finds the index of the first set element.
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::mask32x8;
    /// assert_eq!(mask32x8::splat(false).first_set(), None);
    /// assert_eq!(mask32x8::splat(true).first_set(), Some(0));
    ///
    /// let mask = mask32x8::from_array([false, true, false, false, true, false, false, true]);
    /// assert_eq!(mask.first_set(), Some(1));
    /// ```
    #[inline]
    #[must_use = "method returns the index and does not mutate the original value"]
    pub fn first_set(self) -> Option<usize> {
        // If bitmasks are efficient, using them is better
        if cfg!(target_feature = "sse") && N <= 64 {
            let tz = self.to_bitmask().trailing_zeros();
            return if tz == 64 { None } else { Some(tz as usize) };
        }

        // To find the first set index:
        // * create a vector 0..N
        // * replace unset mask elements in that vector with -1
        // * perform _unsigned_ reduce-min
        // * check if the result is -1 or an index

        let index = Simd::from_array(
            const {
                let mut index = [0; N];
                let mut i = 0;
                while i < N {
                    index[i] = i;
                    i += 1;
                }
                index
            },
        );

        // Safety: the input and output are integer vectors
        let index: Simd<T, N> = unsafe { core::intrinsics::simd::simd_cast(index) };

        let masked_index = self.select(index, Self::splat(true).to_int());

        // Safety: the input and output are integer vectors
        let masked_index: Simd<T::Unsigned, N> =
            unsafe { core::intrinsics::simd::simd_cast(masked_index) };

        // Safety: the input is an integer vector
        let min_index: T::Unsigned =
            unsafe { core::intrinsics::simd::simd_reduce_min(masked_index) };

        // Safety: the return value is the unsigned version of T
        let min_index: T = unsafe { core::mem::transmute_copy(&min_index) };

        if min_index.eq(T::TRUE) {
            None
        } else {
            Some(min_index.to_usize())
        }
    }
}

// vector/array conversion
impl<T, const N: usize> From<[bool; N]> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn from(array: [bool; N]) -> Self {
        Self::from_array(array)
    }
}

impl<T, const N: usize> From<Mask<T, N>> for [bool; N]
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn from(vector: Mask<T, N>) -> Self {
        vector.to_array()
    }
}

impl<T, const N: usize> Default for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    #[must_use = "method returns a defaulted mask with all elements set to false (0)"]
    fn default() -> Self {
        Self::splat(false)
    }
}

impl<T, const N: usize> PartialEq for Mask<T, N>
where
    T: MaskElement + PartialEq,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    #[must_use = "method returns a new bool and does not mutate the original value"]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T, const N: usize> PartialOrd for Mask<T, N>
where
    T: MaskElement + PartialOrd,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    #[must_use = "method returns a new Ordering and does not mutate the original value"]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const N: usize> fmt::Debug for Mask<T, N>
where
    T: MaskElement + fmt::Debug,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..N).map(|i| self.test(i)))
            .finish()
    }
}

impl<T, const N: usize> core::ops::BitAnd for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl<T, const N: usize> core::ops::BitAnd<bool> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitand(self, rhs: bool) -> Self {
        self & Self::splat(rhs)
    }
}

impl<T, const N: usize> core::ops::BitAnd<Mask<T, N>> for bool
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Mask<T, N>;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitand(self, rhs: Mask<T, N>) -> Mask<T, N> {
        Mask::splat(self) & rhs
    }
}

impl<T, const N: usize> core::ops::BitOr for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl<T, const N: usize> core::ops::BitOr<bool> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitor(self, rhs: bool) -> Self {
        self | Self::splat(rhs)
    }
}

impl<T, const N: usize> core::ops::BitOr<Mask<T, N>> for bool
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Mask<T, N>;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitor(self, rhs: Mask<T, N>) -> Mask<T, N> {
        Mask::splat(self) | rhs
    }
}

impl<T, const N: usize> core::ops::BitXor for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<T, const N: usize> core::ops::BitXor<bool> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitxor(self, rhs: bool) -> Self::Output {
        self ^ Self::splat(rhs)
    }
}

impl<T, const N: usize> core::ops::BitXor<Mask<T, N>> for bool
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Mask<T, N>;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn bitxor(self, rhs: Mask<T, N>) -> Self::Output {
        Mask::splat(self) ^ rhs
    }
}

impl<T, const N: usize> core::ops::Not for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Mask<T, N>;
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original value"]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl<T, const N: usize> core::ops::BitAndAssign for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = self.0 & rhs.0;
    }
}

impl<T, const N: usize> core::ops::BitAndAssign<bool> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: bool) {
        *self &= Self::splat(rhs);
    }
}

impl<T, const N: usize> core::ops::BitOrAssign for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = self.0 | rhs.0;
    }
}

impl<T, const N: usize> core::ops::BitOrAssign<bool> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: bool) {
        *self |= Self::splat(rhs);
    }
}

impl<T, const N: usize> core::ops::BitXorAssign for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = self.0 ^ rhs.0;
    }
}

impl<T, const N: usize> core::ops::BitXorAssign<bool> for Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: bool) {
        *self ^= Self::splat(rhs);
    }
}

macro_rules! impl_from {
    { $from:ty  => $($to:ty),* } => {
        $(
        impl<const N: usize> From<Mask<$from, N>> for Mask<$to, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline]
            fn from(value: Mask<$from, N>) -> Self {
                value.cast()
            }
        }
        )*
    }
}
impl_from! { i8 => i16, i32, i64, isize }
impl_from! { i16 => i32, i64, isize, i8 }
impl_from! { i32 => i64, isize, i8, i16 }
impl_from! { i64 => isize, i8, i16, i32 }
impl_from! { isize => i8, i16, i32, i64 }
