mod float;
mod int;
mod uint;

pub use float::*;
pub use int::*;
pub use uint::*;

// Vectors of pointers are not for public use at the current time.
pub(crate) mod ptr;

use crate::simd::intrinsics;
use crate::simd::{LaneCount, Mask, MaskElement, SupportedLaneCount};

/// A SIMD vector of `LANES` elements of type `T`.
#[repr(simd)]
pub struct Simd<T, const LANES: usize>([T; LANES])
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount;

impl<T, const LANES: usize> Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    /// Number of lanes in this vector.
    pub const LANES: usize = LANES;

    /// Get the number of lanes in this vector.
    pub const fn lanes(&self) -> usize {
        LANES
    }

    /// Construct a SIMD vector by setting all lanes to the given value.
    pub const fn splat(value: T) -> Self {
        Self([value; LANES])
    }

    /// Returns an array reference containing the entire SIMD vector.
    pub const fn as_array(&self) -> &[T; LANES] {
        &self.0
    }

    /// Returns a mutable array reference containing the entire SIMD vector.
    pub fn as_mut_array(&mut self) -> &mut [T; LANES] {
        &mut self.0
    }

    /// Converts an array to a SIMD vector.
    pub const fn from_array(array: [T; LANES]) -> Self {
        Self(array)
    }

    /// Converts a SIMD vector to an array.
    pub const fn to_array(self) -> [T; LANES] {
        self.0
    }

    /// Converts a slice to a SIMD vector containing `slice[..LANES]`
    /// # Panics
    /// `from_slice` will panic if the slice's `len` is less than the vector's `Simd::LANES`.
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self {
        assert!(
            slice.len() >= LANES,
            "slice length must be at least the number of lanes"
        );
        let mut array = [slice[0]; LANES];
        let mut i = 0;
        while i < LANES {
            array[i] = slice[i];
            i += 1;
        }
        Self(array)
    }

    /// Performs lanewise conversion of a SIMD vector's elements to another SIMD-valid type.
    /// This follows the semantics of Rust's `as` conversion for casting
    /// integers to unsigned integers (interpreting as the other type, so `-1` to `MAX`),
    /// and from floats to integers (truncating, or saturating at the limits) for each lane,
    /// or vice versa.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::Simd;
    /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
    /// let floats: Simd<f32, 4> = Simd::from_array([1.9, -4.5, f32::INFINITY, f32::NAN]);
    /// let ints = floats.cast::<i32>();
    /// assert_eq!(ints, Simd::from_array([1, -4, i32::MAX, 0]));
    ///
    /// // Formally equivalent, but `Simd::cast` can optimize better.
    /// assert_eq!(ints, Simd::from_array(floats.to_array().map(|x| x as i32)));
    ///
    /// // The float conversion does not round-trip.
    /// let floats_again = ints.cast();
    /// assert_ne!(floats, floats_again);
    /// assert_eq!(floats_again, Simd::from_array([1.0, -4.0, 2147483647.0, 0.0]));
    /// ```
    #[must_use]
    #[inline]
    #[cfg(not(bootstrap))]
    pub fn cast<U: SimdElement>(self) -> Simd<U, LANES> {
        unsafe { intrinsics::simd_as(self) }
    }

    /// Reads from potentially discontiguous indices in `slice` to construct a SIMD vector.
    /// If an index is out-of-bounds, the lane is instead selected from the `or` vector.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::Simd;
    /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    ///
    /// let result = Simd::gather_or(&vec, idxs, alt); // Note the lane that is out-of-bounds.
    /// assert_eq!(result, Simd::from_array([-5, 13, 10, 15]));
    /// ```
    #[must_use]
    #[inline]
    pub fn gather_or(slice: &[T], idxs: Simd<usize, LANES>, or: Self) -> Self {
        Self::gather_select(slice, Mask::splat(true), idxs, or)
    }

    /// Reads from potentially discontiguous indices in `slice` to construct a SIMD vector.
    /// If an index is out-of-bounds, the lane is set to the default value for the type.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::Simd;
    /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);
    ///
    /// let result = Simd::gather_or_default(&vec, idxs); // Note the lane that is out-of-bounds.
    /// assert_eq!(result, Simd::from_array([0, 13, 10, 15]));
    /// ```
    #[must_use]
    #[inline]
    pub fn gather_or_default(slice: &[T], idxs: Simd<usize, LANES>) -> Self
    where
        T: Default,
    {
        Self::gather_or(slice, idxs, Self::splat(T::default()))
    }

    /// Reads from potentially discontiguous indices in `slice` to construct a SIMD vector.
    /// The mask `enable`s all `true` lanes and disables all `false` lanes.
    /// If an index is disabled or is out-of-bounds, the lane is selected from the `or` vector.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::{Simd, Mask};
    /// # #[cfg(not(feature = "std"))] use core::simd::{Simd, Mask};
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    /// let enable = Mask::from_array([true, true, true, false]); // Note the mask of the last lane.
    ///
    /// let result = Simd::gather_select(&vec, enable, idxs, alt); // Note the lane that is out-of-bounds.
    /// assert_eq!(result, Simd::from_array([-5, 13, 10, -2]));
    /// ```
    #[must_use]
    #[inline]
    pub fn gather_select(
        slice: &[T],
        enable: Mask<isize, LANES>,
        idxs: Simd<usize, LANES>,
        or: Self,
    ) -> Self {
        let enable: Mask<isize, LANES> = enable & idxs.lanes_lt(Simd::splat(slice.len()));
        // SAFETY: We have masked-off out-of-bounds lanes.
        unsafe { Self::gather_select_unchecked(slice, enable, idxs, or) }
    }

    /// Reads from potentially discontiguous indices in `slice` to construct a SIMD vector.
    /// The mask `enable`s all `true` lanes and disables all `false` lanes.
    /// If an index is disabled, the lane is selected from the `or` vector.
    ///
    /// # Safety
    ///
    /// Calling this function with an `enable`d out-of-bounds index is *[undefined behavior]*
    /// even if the resulting value is not used.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::{Simd, Mask};
    /// # #[cfg(not(feature = "std"))] use core::simd::{Simd, Mask};
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    /// let enable = Mask::from_array([true, true, true, false]); // Note the final mask lane.
    /// // If this mask was used to gather, it would be unsound. Let's fix that.
    /// let enable = enable & idxs.lanes_lt(Simd::splat(vec.len()));
    ///
    /// // We have masked the OOB lane, so it's safe to gather now.
    /// let result = unsafe { Simd::gather_select_unchecked(&vec, enable, idxs, alt) };
    /// assert_eq!(result, Simd::from_array([-5, 13, 10, -2]));
    /// ```
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[must_use]
    #[inline]
    pub unsafe fn gather_select_unchecked(
        slice: &[T],
        enable: Mask<isize, LANES>,
        idxs: Simd<usize, LANES>,
        or: Self,
    ) -> Self {
        let base_ptr = crate::simd::ptr::SimdConstPtr::splat(slice.as_ptr());
        // Ferris forgive me, I have done pointer arithmetic here.
        let ptrs = base_ptr.wrapping_add(idxs);
        // SAFETY: The ptrs have been bounds-masked to prevent memory-unsafe reads insha'allah
        unsafe { intrinsics::simd_gather(or, ptrs, enable.to_int()) }
    }

    /// Writes the values in a SIMD vector to potentially discontiguous indices in `slice`.
    /// If two lanes in the scattered vector would write to the same index
    /// only the last lane is guaranteed to actually be written.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::Simd;
    /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 0]);
    /// let vals = Simd::from_array([-27, 82, -41, 124]);
    ///
    /// vals.scatter(&mut vec, idxs); // index 0 receives two writes.
    /// assert_eq!(vec, vec![124, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    #[inline]
    pub fn scatter(self, slice: &mut [T], idxs: Simd<usize, LANES>) {
        self.scatter_select(slice, Mask::splat(true), idxs)
    }

    /// Writes the values in a SIMD vector to multiple potentially discontiguous indices in `slice`.
    /// The mask `enable`s all `true` lanes and disables all `false` lanes.
    /// If an enabled index is out-of-bounds, the lane is not written.
    /// If two enabled lanes in the scattered vector would write to the same index,
    /// only the last lane is guaranteed to actually be written.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::{Simd, Mask};
    /// # #[cfg(not(feature = "std"))] use core::simd::{Simd, Mask};
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 0]);
    /// let vals = Simd::from_array([-27, 82, -41, 124]);
    /// let enable = Mask::from_array([true, true, true, false]); // Note the mask of the last lane.
    ///
    /// vals.scatter_select(&mut vec, enable, idxs); // index 0's second write is masked, thus omitted.
    /// assert_eq!(vec, vec![-41, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    #[inline]
    pub fn scatter_select(
        self,
        slice: &mut [T],
        enable: Mask<isize, LANES>,
        idxs: Simd<usize, LANES>,
    ) {
        let enable: Mask<isize, LANES> = enable & idxs.lanes_lt(Simd::splat(slice.len()));
        // SAFETY: We have masked-off out-of-bounds lanes.
        unsafe { self.scatter_select_unchecked(slice, enable, idxs) }
    }

    /// Writes the values in a SIMD vector to multiple potentially discontiguous indices in `slice`.
    /// The mask `enable`s all `true` lanes and disables all `false` lanes.
    /// If two enabled lanes in the scattered vector would write to the same index,
    /// only the last lane is guaranteed to actually be written.
    ///
    /// # Safety
    ///
    /// Calling this function with an enabled out-of-bounds index is *[undefined behavior]*,
    /// and may lead to memory corruption.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "std")] use core_simd::{Simd, Mask};
    /// # #[cfg(not(feature = "std"))] use core::simd::{Simd, Mask};
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 0]);
    /// let vals = Simd::from_array([-27, 82, -41, 124]);
    /// let enable = Mask::from_array([true, true, true, false]); // Note the mask of the last lane.
    /// // If this mask was used to scatter, it would be unsound. Let's fix that.
    /// let enable = enable & idxs.lanes_lt(Simd::splat(vec.len()));
    ///
    /// // We have masked the OOB lane, so it's safe to scatter now.
    /// unsafe { vals.scatter_select_unchecked(&mut vec, enable, idxs); }
    /// // index 0's second write is masked, thus was omitted.
    /// assert_eq!(vec, vec![-41, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub unsafe fn scatter_select_unchecked(
        self,
        slice: &mut [T],
        enable: Mask<isize, LANES>,
        idxs: Simd<usize, LANES>,
    ) {
        // SAFETY: This block works with *mut T derived from &mut 'a [T],
        // which means it is delicate in Rust's borrowing model, circa 2021:
        // &mut 'a [T] asserts uniqueness, so deriving &'a [T] invalidates live *mut Ts!
        // Even though this block is largely safe methods, it must be exactly this way
        // to prevent invalidating the raw ptrs while they're live.
        // Thus, entering this block requires all values to use being already ready:
        // 0. idxs we want to write to, which are used to construct the mask.
        // 1. enable, which depends on an initial &'a [T] and the idxs.
        // 2. actual values to scatter (self).
        // 3. &mut [T] which will become our base ptr.
        unsafe {
            // Now Entering ☢️ *mut T Zone
            let base_ptr = crate::simd::ptr::SimdMutPtr::splat(slice.as_mut_ptr());
            // Ferris forgive me, I have done pointer arithmetic here.
            let ptrs = base_ptr.wrapping_add(idxs);
            // The ptrs have been bounds-masked to prevent memory-unsafe writes insha'allah
            intrinsics::simd_scatter(self, ptrs, enable.to_int())
            // Cleared ☢️ *mut T Zone
        }
    }
}

impl<T, const LANES: usize> Copy for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<T, const LANES: usize> Clone for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const LANES: usize> Default for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + Default,
{
    #[inline]
    fn default() -> Self {
        Self::splat(T::default())
    }
}

impl<T, const LANES: usize> PartialEq for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // TODO use SIMD equality
        self.to_array() == other.to_array()
    }
}

impl<T, const LANES: usize> PartialOrd for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        // TODO use SIMD equality
        self.to_array().partial_cmp(other.as_ref())
    }
}

impl<T, const LANES: usize> Eq for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + Eq,
{
}

impl<T, const LANES: usize> Ord for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // TODO use SIMD equality
        self.to_array().cmp(other.as_ref())
    }
}

impl<T, const LANES: usize> core::hash::Hash for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + core::hash::Hash,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: core::hash::Hasher,
    {
        self.as_array().hash(state)
    }
}

// array references
impl<T, const LANES: usize> AsRef<[T; LANES]> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_ref(&self) -> &[T; LANES] {
        &self.0
    }
}

impl<T, const LANES: usize> AsMut<[T; LANES]> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T; LANES] {
        &mut self.0
    }
}

// slice references
impl<T, const LANES: usize> AsRef<[T]> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T, const LANES: usize> AsMut<[T]> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

// vector/array conversion
impl<T, const LANES: usize> From<[T; LANES]> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    fn from(array: [T; LANES]) -> Self {
        Self(array)
    }
}

impl<T, const LANES: usize> From<Simd<T, LANES>> for [T; LANES]
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    fn from(vector: Simd<T, LANES>) -> Self {
        vector.to_array()
    }
}

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Marker trait for types that may be used as SIMD vector elements.
/// SAFETY: This trait, when implemented, asserts the compiler can monomorphize
/// `#[repr(simd)]` structs with the marked type as an element.
/// Strictly, it is valid to impl if the vector will not be miscompiled.
/// Practically, it is user-unfriendly to impl it if the vector won't compile,
/// even when no soundness guarantees are broken by allowing the user to try.
pub unsafe trait SimdElement: Sealed + Copy {
    /// The mask element type corresponding to this element type.
    type Mask: MaskElement;
}

impl Sealed for u8 {}
unsafe impl SimdElement for u8 {
    type Mask = i8;
}

impl Sealed for u16 {}
unsafe impl SimdElement for u16 {
    type Mask = i16;
}

impl Sealed for u32 {}
unsafe impl SimdElement for u32 {
    type Mask = i32;
}

impl Sealed for u64 {}
unsafe impl SimdElement for u64 {
    type Mask = i64;
}

impl Sealed for usize {}
unsafe impl SimdElement for usize {
    type Mask = isize;
}

impl Sealed for i8 {}
unsafe impl SimdElement for i8 {
    type Mask = i8;
}

impl Sealed for i16 {}
unsafe impl SimdElement for i16 {
    type Mask = i16;
}

impl Sealed for i32 {}
unsafe impl SimdElement for i32 {
    type Mask = i32;
}

impl Sealed for i64 {}
unsafe impl SimdElement for i64 {
    type Mask = i64;
}

impl Sealed for isize {}
unsafe impl SimdElement for isize {
    type Mask = isize;
}

impl Sealed for f32 {}
unsafe impl SimdElement for f32 {
    type Mask = i32;
}

impl Sealed for f64 {}
unsafe impl SimdElement for f64 {
    type Mask = i64;
}
