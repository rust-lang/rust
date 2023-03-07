mod float;
mod int;
mod uint;

pub use float::*;
pub use int::*;
pub use uint::*;

// Vectors of pointers are not for public use at the current time.
pub(crate) mod ptr;

use crate::simd::{
    intrinsics, LaneCount, Mask, MaskElement, SimdPartialOrd, SupportedLaneCount, Swizzle,
};

/// A SIMD vector of `LANES` elements of type `T`. `Simd<T, N>` has the same shape as [`[T; N]`](array), but operates like `T`.
///
/// Two vectors of the same type and length will, by convention, support the operators (+, *, etc.) that `T` does.
/// These take the lanes at each index on the left-hand side and right-hand side, perform the operation,
/// and return the result in the same lane in a vector of equal size. For a given operator, this is equivalent to zipping
/// the two arrays together and mapping the operator over each lane.
///
/// ```rust
/// # #![feature(array_zip, portable_simd)]
/// # use core::simd::{Simd};
/// let a0: [i32; 4] = [-2, 0, 2, 4];
/// let a1 = [10, 9, 8, 7];
/// let zm_add = a0.zip(a1).map(|(lhs, rhs)| lhs + rhs);
/// let zm_mul = a0.zip(a1).map(|(lhs, rhs)| lhs * rhs);
///
/// // `Simd<T, N>` implements `From<[T; N]>`
/// let (v0, v1) = (Simd::from(a0), Simd::from(a1));
/// // Which means arrays implement `Into<Simd<T, N>>`.
/// assert_eq!(v0 + v1, zm_add.into());
/// assert_eq!(v0 * v1, zm_mul.into());
/// ```
///
/// `Simd` with integers has the quirk that these operations are also inherently wrapping, as if `T` was [`Wrapping<T>`].
/// Thus, `Simd` does not implement `wrapping_add`, because that is the default behavior.
/// This means there is no warning on overflows, even in "debug" builds.
/// For most applications where `Simd` is appropriate, it is "not a bug" to wrap,
/// and even "debug builds" are unlikely to tolerate the loss of performance.
/// You may want to consider using explicitly checked arithmetic if such is required.
/// Division by zero still causes a panic, so you may want to consider using floating point numbers if that is unacceptable.
///
/// [`Wrapping<T>`]: core::num::Wrapping
///
/// # Layout
/// `Simd<T, N>` has a layout similar to `[T; N]` (identical "shapes"), but with a greater alignment.
/// `[T; N]` is aligned to `T`, but `Simd<T, N>` will have an alignment based on both `T` and `N`.
/// It is thus sound to [`transmute`] `Simd<T, N>` to `[T; N]`, and will typically optimize to zero cost,
/// but the reverse transmutation is more likely to require a copy the compiler cannot simply elide.
///
/// # ABI "Features"
/// Due to Rust's safety guarantees, `Simd<T, N>` is currently passed to and from functions via memory, not SIMD registers,
/// except as an optimization. `#[inline]` hints are recommended on functions that accept `Simd<T, N>` or return it.
/// The need for this may be corrected in the future.
///
/// # Safe SIMD with Unsafe Rust
///
/// Operations with `Simd` are typically safe, but there are many reasons to want to combine SIMD with `unsafe` code.
/// Care must be taken to respect differences between `Simd` and other types it may be transformed into or derived from.
/// In particular, the layout of `Simd<T, N>` may be similar to `[T; N]`, and may allow some transmutations,
/// but references to `[T; N]` are not interchangeable with those to `Simd<T, N>`.
/// Thus, when using `unsafe` Rust to read and write `Simd<T, N>` through [raw pointers], it is a good idea to first try with
/// [`read_unaligned`] and [`write_unaligned`]. This is because:
/// - [`read`] and [`write`] require full alignment (in this case, `Simd<T, N>`'s alignment)
/// - the likely source for reading or destination for writing `Simd<T, N>` is [`[T]`](slice) and similar types, aligned to `T`
/// - combining these actions would violate the `unsafe` contract and explode the program into a puff of **undefined behavior**
/// - the compiler can implicitly adjust layouts to make unaligned reads or writes fully aligned if it sees the optimization
/// - most contemporary processors suffer no performance penalty for "unaligned" reads and writes that are aligned at runtime
///
/// By imposing less obligations, unaligned functions are less likely to make the program unsound,
/// and may be just as fast as stricter alternatives.
/// When trying to guarantee alignment, [`[T]::as_simd`][as_simd] is an option for converting `[T]` to `[Simd<T, N>]`,
/// and allows soundly operating on an aligned SIMD body, but it may cost more time when handling the scalar head and tail.
/// If these are not sufficient, then it is most ideal to design data structures to be already aligned
/// to the `Simd<T, N>` you wish to use before using `unsafe` Rust to read or write.
/// More conventional ways to compensate for these facts, like materializing `Simd` to or from an array first,
/// are handled by safe methods like [`Simd::from_array`] and [`Simd::from_slice`].
///
/// [`transmute`]: core::mem::transmute
/// [raw pointers]: pointer
/// [`read_unaligned`]: pointer::read_unaligned
/// [`write_unaligned`]: pointer::write_unaligned
/// [`read`]: pointer::read
/// [`write`]: pointer::write
/// [as_simd]: slice::as_simd
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

    /// Returns the number of lanes in this SIMD vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::u32x4;
    /// let v = u32x4::splat(0);
    /// assert_eq!(v.lanes(), 4);
    /// ```
    pub const fn lanes(&self) -> usize {
        LANES
    }

    /// Constructs a new SIMD vector with all lanes set to the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::u32x4;
    /// let v = u32x4::splat(8);
    /// assert_eq!(v.as_array(), &[8, 8, 8, 8]);
    /// ```
    pub fn splat(value: T) -> Self {
        // This is preferred over `[value; LANES]`, since it's explicitly a splat:
        // https://github.com/rust-lang/rust/issues/97804
        struct Splat;
        impl<const LANES: usize> Swizzle<1, LANES> for Splat {
            const INDEX: [usize; LANES] = [0; LANES];
        }
        Splat::swizzle(Simd::<T, 1>::from([value]))
    }

    /// Returns an array reference containing the entire SIMD vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::{Simd, u64x4};
    /// let v: u64x4 = Simd::from_array([0, 1, 2, 3]);
    /// assert_eq!(v.as_array(), &[0, 1, 2, 3]);
    /// ```
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

    /// Converts a slice to a SIMD vector containing `slice[..LANES]`.
    ///
    /// # Panics
    ///
    /// Panics if the slice's length is less than the vector's `Simd::LANES`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::u32x4;
    /// let source = vec![1, 2, 3, 4, 5, 6];
    /// let v = u32x4::from_slice(&source);
    /// assert_eq!(v.as_array(), &[1, 2, 3, 4]);
    /// ```
    #[must_use]
    pub const fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() >= LANES, "slice length must be at least the number of lanes");
        let mut array = [slice[0]; LANES];
        let mut i = 0;
        while i < LANES {
            array[i] = slice[i];
            i += 1;
        }
        Self(array)
    }

    /// Performs lanewise conversion of a SIMD vector's elements to another SIMD-valid type.
    ///
    /// This follows the semantics of Rust's `as` conversion for casting
    /// integers to unsigned integers (interpreting as the other type, so `-1` to `MAX`),
    /// and from floats to integers (truncating, or saturating at the limits) for each lane,
    /// or vice versa.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Simd;
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
    pub fn cast<U: SimdElement>(self) -> Simd<U, LANES> {
        // Safety: The input argument is a vector of a valid SIMD element type.
        unsafe { intrinsics::simd_as(self) }
    }

    /// Rounds toward zero and converts to the same-width integer type, assuming that
    /// the value is finite and fits in that type.
    ///
    /// # Safety
    /// The value must:
    ///
    /// * Not be NaN
    /// * Not be infinite
    /// * Be representable in the return type, after truncating off its fractional part
    ///
    /// If these requirements are infeasible or costly, consider using the safe function [cast],
    /// which saturates on conversion.
    ///
    /// [cast]: Simd::cast
    #[inline]
    pub unsafe fn to_int_unchecked<I>(self) -> Simd<I, LANES>
    where
        T: core::convert::FloatToInt<I>,
        I: SimdElement,
    {
        // Safety: `self` is a vector, and `FloatToInt` ensures the type can be casted to
        // an integer.
        unsafe { intrinsics::simd_cast(self) }
    }

    /// Reads from potentially discontiguous indices in `slice` to construct a SIMD vector.
    /// If an index is out-of-bounds, the lane is instead selected from the `or` vector.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Simd;
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
    /// # use core::simd::Simd;
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
    /// # use core::simd::{Simd, Mask};
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
        let enable: Mask<isize, LANES> = enable & idxs.simd_lt(Simd::splat(slice.len()));
        // Safety: We have masked-off out-of-bounds lanes.
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
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Simd, SimdPartialOrd, Mask};
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    /// let enable = Mask::from_array([true, true, true, false]); // Note the final mask lane.
    /// // If this mask was used to gather, it would be unsound. Let's fix that.
    /// let enable = enable & idxs.simd_lt(Simd::splat(vec.len()));
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
        // Safety: The ptrs have been bounds-masked to prevent memory-unsafe reads insha'allah
        unsafe { intrinsics::simd_gather(or, ptrs, enable.to_int()) }
    }

    /// Writes the values in a SIMD vector to potentially discontiguous indices in `slice`.
    /// If two lanes in the scattered vector would write to the same index
    /// only the last lane is guaranteed to actually be written.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Simd;
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
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Simd, Mask};
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
        let enable: Mask<isize, LANES> = enable & idxs.simd_lt(Simd::splat(slice.len()));
        // Safety: We have masked-off out-of-bounds lanes.
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
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Simd, SimdPartialOrd, Mask};
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 0]);
    /// let vals = Simd::from_array([-27, 82, -41, 124]);
    /// let enable = Mask::from_array([true, true, true, false]); // Note the mask of the last lane.
    /// // If this mask was used to scatter, it would be unsound. Let's fix that.
    /// let enable = enable & idxs.simd_lt(Simd::splat(vec.len()));
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
        // Safety: This block works with *mut T derived from &mut 'a [T],
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
        // Safety: All SIMD vectors are SimdPartialEq, and the comparison produces a valid mask.
        let mask = unsafe {
            let tfvec: Simd<<T as SimdElement>::Mask, LANES> = intrinsics::simd_eq(*self, *other);
            Mask::from_int_unchecked(tfvec)
        };

        // Two vectors are equal if all lanes tested true for vertical equality.
        mask.all()
    }

    #[allow(clippy::partialeq_ne_impl)]
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        // Safety: All SIMD vectors are SimdPartialEq, and the comparison produces a valid mask.
        let mask = unsafe {
            let tfvec: Simd<<T as SimdElement>::Mask, LANES> = intrinsics::simd_ne(*self, *other);
            Mask::from_int_unchecked(tfvec)
        };

        // Two vectors are non-equal if any lane tested true for vertical non-equality.
        mask.any()
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
///
/// # Safety
/// This trait, when implemented, asserts the compiler can monomorphize
/// `#[repr(simd)]` structs with the marked type as an element.
/// Strictly, it is valid to impl if the vector will not be miscompiled.
/// Practically, it is user-unfriendly to impl it if the vector won't compile,
/// even when no soundness guarantees are broken by allowing the user to try.
pub unsafe trait SimdElement: Sealed + Copy {
    /// The mask element type corresponding to this element type.
    type Mask: MaskElement;
}

impl Sealed for u8 {}

// Safety: u8 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for u8 {
    type Mask = i8;
}

impl Sealed for u16 {}

// Safety: u16 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for u16 {
    type Mask = i16;
}

impl Sealed for u32 {}

// Safety: u32 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for u32 {
    type Mask = i32;
}

impl Sealed for u64 {}

// Safety: u64 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for u64 {
    type Mask = i64;
}

impl Sealed for usize {}

// Safety: usize is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for usize {
    type Mask = isize;
}

impl Sealed for i8 {}

// Safety: i8 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for i8 {
    type Mask = i8;
}

impl Sealed for i16 {}

// Safety: i16 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for i16 {
    type Mask = i16;
}

impl Sealed for i32 {}

// Safety: i32 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for i32 {
    type Mask = i32;
}

impl Sealed for i64 {}

// Safety: i64 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for i64 {
    type Mask = i64;
}

impl Sealed for isize {}

// Safety: isize is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for isize {
    type Mask = isize;
}

impl Sealed for f32 {}

// Safety: f32 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for f32 {
    type Mask = i32;
}

impl Sealed for f64 {}

// Safety: f64 is a valid SIMD element type, and is supported by this API
unsafe impl SimdElement for f64 {
    type Mask = i64;
}
