use crate::simd::{
    intrinsics, LaneCount, Mask, MaskElement, SimdConstPtr, SimdMutPtr, SimdPartialOrd,
    SupportedLaneCount, Swizzle,
};
use core::convert::{TryFrom, TryInto};

/// A SIMD vector with the shape of `[T; N]` but the operations of `T`.
///
/// `Simd<T, N>` supports the operators (+, *, etc.) that `T` does in "elementwise" fashion.
/// These take the element at each index from the left-hand side and right-hand side,
/// perform the operation, then return the result in the same index in a vector of equal size.
/// However, `Simd` differs from normal iteration and normal arrays:
/// - `Simd<T, N>` executes `N` operations in a single step with no `break`s
/// - `Simd<T, N>` can have an alignment greater than `T`, for better mechanical sympathy
///
/// By always imposing these constraints on `Simd`, it is easier to compile elementwise operations
/// into machine instructions that can themselves be executed in parallel.
///
/// ```rust
/// # #![feature(portable_simd)]
/// # use core::simd::{Simd};
/// # use core::array;
/// let a: [i32; 4] = [-2, 0, 2, 4];
/// let b = [10, 9, 8, 7];
/// let sum = array::from_fn(|i| a[i] + b[i]);
/// let prod = array::from_fn(|i| a[i] * b[i]);
///
/// // `Simd<T, N>` implements `From<[T; N]>`
/// let (v, w) = (Simd::from(a), Simd::from(b));
/// // Which means arrays implement `Into<Simd<T, N>>`.
/// assert_eq!(v + w, sum.into());
/// assert_eq!(v * w, prod.into());
/// ```
///
///
/// `Simd` with integer elements treats operators as wrapping, as if `T` was [`Wrapping<T>`].
/// Thus, `Simd` does not implement `wrapping_add`, because that is the default behavior.
/// This means there is no warning on overflows, even in "debug" builds.
/// For most applications where `Simd` is appropriate, it is "not a bug" to wrap,
/// and even "debug builds" are unlikely to tolerate the loss of performance.
/// You may want to consider using explicitly checked arithmetic if such is required.
/// Division by zero on integers still causes a panic, so
/// you may want to consider using `f32` or `f64` if that is unacceptable.
///
/// [`Wrapping<T>`]: core::num::Wrapping
///
/// # Layout
/// `Simd<T, N>` has a layout similar to `[T; N]` (identical "shapes"), with a greater alignment.
/// `[T; N]` is aligned to `T`, but `Simd<T, N>` will have an alignment based on both `T` and `N`.
/// Thus it is sound to [`transmute`] `Simd<T, N>` to `[T; N]` and should optimize to "zero cost",
/// but the reverse transmutation may require a copy the compiler cannot simply elide.
///
/// # ABI "Features"
/// Due to Rust's safety guarantees, `Simd<T, N>` is currently passed and returned via memory,
/// not SIMD registers, except as an optimization. Using `#[inline]` on functions that accept
/// `Simd<T, N>` or return it is recommended, at the cost of code generation time, as
/// inlining SIMD-using functions can omit a large function prolog or epilog and thus
/// improve both speed and code size. The need for this may be corrected in the future.
///
/// Using `#[inline(always)]` still requires additional care.
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
/// - `Simd<T, N>` is often read from or written to [`[T]`](slice) and other types aligned to `T`
/// - combining these actions violates the `unsafe` contract and explodes the program into
///   a puff of **undefined behavior**
/// - the compiler can implicitly adjust layouts to make unaligned reads or writes fully aligned
///   if it sees the optimization
/// - most contemporary processors with "aligned" and "unaligned" read and write instructions
///   exhibit no performance difference if the "unaligned" variant is aligned at runtime
///
/// Less obligations mean unaligned reads and writes are less likely to make the program unsound,
/// and may be just as fast as stricter alternatives.
/// When trying to guarantee alignment, [`[T]::as_simd`][as_simd] is an option for
/// converting `[T]` to `[Simd<T, N>]`, and allows soundly operating on an aligned SIMD body,
/// but it may cost more time when handling the scalar head and tail.
/// If these are not enough, it is most ideal to design data structures to be already aligned
/// to `mem::align_of::<Simd<T, N>>()` before using `unsafe` Rust to read or write.
/// Other ways to compensate for these facts, like materializing `Simd` to or from an array first,
/// are handled by safe methods like [`Simd::from_array`] and [`Simd::from_slice`].
///
/// [`transmute`]: core::mem::transmute
/// [raw pointers]: pointer
/// [`read_unaligned`]: pointer::read_unaligned
/// [`write_unaligned`]: pointer::write_unaligned
/// [`read`]: pointer::read
/// [`write`]: pointer::write
/// [as_simd]: slice::as_simd
//
// NOTE: Accessing the inner array directly in any way (e.g. by using the `.0` field syntax) or
// directly constructing an instance of the type (i.e. `let vector = Simd(array)`) should be
// avoided, as it will likely become illegal on `#[repr(simd)]` structs in the future. It also
// causes rustc to emit illegal LLVM IR in some cases.
#[repr(simd)]
pub struct Simd<T, const N: usize>([T; N])
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement;

impl<T, const N: usize> Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    /// Number of elements in this vector.
    pub const LANES: usize = N;

    /// Returns the number of elements in this SIMD vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::u32x4;
    /// let v = u32x4::splat(0);
    /// assert_eq!(v.lanes(), 4);
    /// ```
    #[inline]
    pub const fn lanes(&self) -> usize {
        Self::LANES
    }

    /// Constructs a new SIMD vector with all elements set to the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::u32x4;
    /// let v = u32x4::splat(8);
    /// assert_eq!(v.as_array(), &[8, 8, 8, 8]);
    /// ```
    #[inline]
    pub fn splat(value: T) -> Self {
        // This is preferred over `[value; N]`, since it's explicitly a splat:
        // https://github.com/rust-lang/rust/issues/97804
        struct Splat;
        impl<const N: usize> Swizzle<1, N> for Splat {
            const INDEX: [usize; N] = [0; N];
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
    #[inline]
    pub const fn as_array(&self) -> &[T; N] {
        // SAFETY: `Simd<T, N>` is just an overaligned `[T; N]` with
        // potential padding at the end, so pointer casting to a
        // `&[T; N]` is safe.
        //
        // NOTE: This deliberately doesn't just use `&self.0`, see the comment
        // on the struct definition for details.
        unsafe { &*(self as *const Self as *const [T; N]) }
    }

    /// Returns a mutable array reference containing the entire SIMD vector.
    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [T; N] {
        // SAFETY: `Simd<T, N>` is just an overaligned `[T; N]` with
        // potential padding at the end, so pointer casting to a
        // `&mut [T; N]` is safe.
        //
        // NOTE: This deliberately doesn't just use `&mut self.0`, see the comment
        // on the struct definition for details.
        unsafe { &mut *(self as *mut Self as *mut [T; N]) }
    }

    /// Load a vector from an array of `T`.
    ///
    /// This function is necessary since `repr(simd)` has padding for non-power-of-2 vectors (at the time of writing).
    /// With padding, `read_unaligned` will read past the end of an array of N elements.
    ///
    /// # Safety
    /// Reading `ptr` must be safe, as if by `<*const [T; N]>::read_unaligned`.
    #[inline]
    const unsafe fn load(ptr: *const [T; N]) -> Self {
        // There are potentially simpler ways to write this function, but this should result in
        // LLVM `load <N x T>`

        let mut tmp = core::mem::MaybeUninit::<Self>::uninit();
        // SAFETY: `Simd<T, N>` always contains `N` elements of type `T`.  It may have padding
        // which does not need to be initialized.  The safety of reading `ptr` is ensured by the
        // caller.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr().cast(), 1);
            tmp.assume_init()
        }
    }

    /// Store a vector to an array of `T`.
    ///
    /// See `load` as to why this function is necessary.
    ///
    /// # Safety
    /// Writing to `ptr` must be safe, as if by `<*mut [T; N]>::write_unaligned`.
    #[inline]
    const unsafe fn store(self, ptr: *mut [T; N]) {
        // There are potentially simpler ways to write this function, but this should result in
        // LLVM `store <N x T>`

        // Creating a temporary helps LLVM turn the memcpy into a store.
        let tmp = self;
        // SAFETY: `Simd<T, N>` always contains `N` elements of type `T`.  The safety of writing
        // `ptr` is ensured by the caller.
        unsafe { core::ptr::copy_nonoverlapping(tmp.as_array(), ptr, 1) }
    }

    /// Converts an array to a SIMD vector.
    #[inline]
    pub const fn from_array(array: [T; N]) -> Self {
        // SAFETY: `&array` is safe to read.
        //
        // FIXME: We currently use a pointer load instead of `transmute_copy` because `repr(simd)`
        // results in padding for non-power-of-2 vectors (so vectors are larger than arrays).
        //
        // NOTE: This deliberately doesn't just use `Self(array)`, see the comment
        // on the struct definition for details.
        unsafe { Self::load(&array) }
    }

    /// Converts a SIMD vector to an array.
    #[inline]
    pub const fn to_array(self) -> [T; N] {
        let mut tmp = core::mem::MaybeUninit::uninit();
        // SAFETY: writing to `tmp` is safe and initializes it.
        //
        // FIXME: We currently use a pointer store instead of `transmute_copy` because `repr(simd)`
        // results in padding for non-power-of-2 vectors (so vectors are larger than arrays).
        //
        // NOTE: This deliberately doesn't just use `self.0`, see the comment
        // on the struct definition for details.
        unsafe {
            self.store(tmp.as_mut_ptr());
            tmp.assume_init()
        }
    }

    /// Converts a slice to a SIMD vector containing `slice[..N]`.
    ///
    /// # Panics
    ///
    /// Panics if the slice's length is less than the vector's `Simd::N`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::u32x4;
    /// let source = vec![1, 2, 3, 4, 5, 6];
    /// let v = u32x4::from_slice(&source);
    /// assert_eq!(v.as_array(), &[1, 2, 3, 4]);
    /// ```
    #[must_use]
    #[inline]
    #[track_caller]
    pub const fn from_slice(slice: &[T]) -> Self {
        assert!(
            slice.len() >= Self::LANES,
            "slice length must be at least the number of elements"
        );
        // SAFETY: We just checked that the slice contains
        // at least `N` elements.
        unsafe { Self::load(slice.as_ptr().cast()) }
    }

    /// Writes a SIMD vector to the first `N` elements of a slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice's length is less than the vector's `Simd::N`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::u32x4;
    /// let mut dest = vec![0; 6];
    /// let v = u32x4::from_array([1, 2, 3, 4]);
    /// v.copy_to_slice(&mut dest);
    /// assert_eq!(&dest, &[1, 2, 3, 4, 0, 0]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn copy_to_slice(self, slice: &mut [T]) {
        assert!(
            slice.len() >= Self::LANES,
            "slice length must be at least the number of elements"
        );
        // SAFETY: We just checked that the slice contains
        // at least `N` elements.
        unsafe { self.store(slice.as_mut_ptr().cast()) }
    }

    /// Reads from potentially discontiguous indices in `slice` to construct a SIMD vector.
    /// If an index is out-of-bounds, the element is instead selected from the `or` vector.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Simd;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);  // Note the index that is out-of-bounds
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    ///
    /// let result = Simd::gather_or(&vec, idxs, alt);
    /// assert_eq!(result, Simd::from_array([-5, 13, 10, 15]));
    /// ```
    #[must_use]
    #[inline]
    pub fn gather_or(slice: &[T], idxs: Simd<usize, N>, or: Self) -> Self {
        Self::gather_select(slice, Mask::splat(true), idxs, or)
    }

    /// Reads from indices in `slice` to construct a SIMD vector.
    /// If an index is out-of-bounds, the element is set to the default given by `T: Default`.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Simd;
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]);  // Note the index that is out-of-bounds
    ///
    /// let result = Simd::gather_or_default(&vec, idxs);
    /// assert_eq!(result, Simd::from_array([0, 13, 10, 15]));
    /// ```
    #[must_use]
    #[inline]
    pub fn gather_or_default(slice: &[T], idxs: Simd<usize, N>) -> Self
    where
        T: Default,
    {
        Self::gather_or(slice, idxs, Self::splat(T::default()))
    }

    /// Reads from indices in `slice` to construct a SIMD vector.
    /// The mask `enable`s all `true` indices and disables all `false` indices.
    /// If an index is disabled or is out-of-bounds, the element is selected from the `or` vector.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::{Simd, Mask};
    /// let vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 5]); // Includes an out-of-bounds index
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    /// let enable = Mask::from_array([true, true, true, false]); // Includes a masked element
    ///
    /// let result = Simd::gather_select(&vec, enable, idxs, alt);
    /// assert_eq!(result, Simd::from_array([-5, 13, 10, -2]));
    /// ```
    #[must_use]
    #[inline]
    pub fn gather_select(
        slice: &[T],
        enable: Mask<isize, N>,
        idxs: Simd<usize, N>,
        or: Self,
    ) -> Self {
        let enable: Mask<isize, N> = enable & idxs.simd_lt(Simd::splat(slice.len()));
        // Safety: We have masked-off out-of-bounds indices.
        unsafe { Self::gather_select_unchecked(slice, enable, idxs, or) }
    }

    /// Reads from indices in `slice` to construct a SIMD vector.
    /// The mask `enable`s all `true` indices and disables all `false` indices.
    /// If an index is disabled, the element is selected from the `or` vector.
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
    /// let idxs = Simd::from_array([9, 3, 0, 5]); // Includes an out-of-bounds index
    /// let alt = Simd::from_array([-5, -4, -3, -2]);
    /// let enable = Mask::from_array([true, true, true, false]); // Includes a masked element
    /// // If this mask was used to gather, it would be unsound. Let's fix that.
    /// let enable = enable & idxs.simd_lt(Simd::splat(vec.len()));
    ///
    /// // The out-of-bounds index has been masked, so it's safe to gather now.
    /// let result = unsafe { Simd::gather_select_unchecked(&vec, enable, idxs, alt) };
    /// assert_eq!(result, Simd::from_array([-5, 13, 10, -2]));
    /// ```
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[must_use]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn gather_select_unchecked(
        slice: &[T],
        enable: Mask<isize, N>,
        idxs: Simd<usize, N>,
        or: Self,
    ) -> Self {
        let base_ptr = Simd::<*const T, N>::splat(slice.as_ptr());
        // Ferris forgive me, I have done pointer arithmetic here.
        let ptrs = base_ptr.wrapping_add(idxs);
        // Safety: The caller is responsible for determining the indices are okay to read
        unsafe { Self::gather_select_ptr(ptrs, enable, or) }
    }

    /// Read elementwise from pointers into a SIMD vector.
    ///
    /// # Safety
    ///
    /// Each read must satisfy the same conditions as [`core::ptr::read`].
    ///
    /// # Example
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Simd, SimdConstPtr};
    /// let values = [6, 2, 4, 9];
    /// let offsets = Simd::from_array([1, 0, 0, 3]);
    /// let source = Simd::splat(values.as_ptr()).wrapping_add(offsets);
    /// let gathered = unsafe { Simd::gather_ptr(source) };
    /// assert_eq!(gathered, Simd::from_array([2, 6, 6, 9]));
    /// ```
    #[must_use]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn gather_ptr(source: Simd<*const T, N>) -> Self
    where
        T: Default,
    {
        // TODO: add an intrinsic that doesn't use a passthru vector, and remove the T: Default bound
        // Safety: The caller is responsible for upholding all invariants
        unsafe { Self::gather_select_ptr(source, Mask::splat(true), Self::default()) }
    }

    /// Conditionally read elementwise from pointers into a SIMD vector.
    /// The mask `enable`s all `true` pointers and disables all `false` pointers.
    /// If a pointer is disabled, the element is selected from the `or` vector,
    /// and no read is performed.
    ///
    /// # Safety
    ///
    /// Enabled elements must satisfy the same conditions as [`core::ptr::read`].
    ///
    /// # Example
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Mask, Simd, SimdConstPtr};
    /// let values = [6, 2, 4, 9];
    /// let enable = Mask::from_array([true, true, false, true]);
    /// let offsets = Simd::from_array([1, 0, 0, 3]);
    /// let source = Simd::splat(values.as_ptr()).wrapping_add(offsets);
    /// let gathered = unsafe { Simd::gather_select_ptr(source, enable, Simd::splat(0)) };
    /// assert_eq!(gathered, Simd::from_array([2, 6, 0, 9]));
    /// ```
    #[must_use]
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn gather_select_ptr(
        source: Simd<*const T, N>,
        enable: Mask<isize, N>,
        or: Self,
    ) -> Self {
        // Safety: The caller is responsible for upholding all invariants
        unsafe { intrinsics::simd_gather(or, source, enable.to_int()) }
    }

    /// Writes the values in a SIMD vector to potentially discontiguous indices in `slice`.
    /// If an index is out-of-bounds, the write is suppressed without panicking.
    /// If two elements in the scattered vector would write to the same index
    /// only the last element is guaranteed to actually be written.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Simd;
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 0]); // Note the duplicate index.
    /// let vals = Simd::from_array([-27, 82, -41, 124]);
    ///
    /// vals.scatter(&mut vec, idxs); // two logical writes means the last wins.
    /// assert_eq!(vec, vec![124, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    #[inline]
    pub fn scatter(self, slice: &mut [T], idxs: Simd<usize, N>) {
        self.scatter_select(slice, Mask::splat(true), idxs)
    }

    /// Writes values from a SIMD vector to multiple potentially discontiguous indices in `slice`.
    /// The mask `enable`s all `true` indices and disables all `false` indices.
    /// If an enabled index is out-of-bounds, the write is suppressed without panicking.
    /// If two enabled elements in the scattered vector would write to the same index,
    /// only the last element is guaranteed to actually be written.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Simd, Mask};
    /// let mut vec: Vec<i32> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// let idxs = Simd::from_array([9, 3, 0, 0]); // Includes an out-of-bounds index
    /// let vals = Simd::from_array([-27, 82, -41, 124]);
    /// let enable = Mask::from_array([true, true, true, false]); // Includes a masked element
    ///
    /// vals.scatter_select(&mut vec, enable, idxs); // The last write is masked, thus omitted.
    /// assert_eq!(vec, vec![-41, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    #[inline]
    pub fn scatter_select(self, slice: &mut [T], enable: Mask<isize, N>, idxs: Simd<usize, N>) {
        let enable: Mask<isize, N> = enable & idxs.simd_lt(Simd::splat(slice.len()));
        // Safety: We have masked-off out-of-bounds indices.
        unsafe { self.scatter_select_unchecked(slice, enable, idxs) }
    }

    /// Writes values from a SIMD vector to multiple potentially discontiguous indices in `slice`.
    /// The mask `enable`s all `true` indices and disables all `false` indices.
    /// If two enabled elements in the scattered vector would write to the same index,
    /// only the last element is guaranteed to actually be written.
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
    /// let enable = Mask::from_array([true, true, true, false]); // Masks the final index
    /// // If this mask was used to scatter, it would be unsound. Let's fix that.
    /// let enable = enable & idxs.simd_lt(Simd::splat(vec.len()));
    ///
    /// // We have masked the OOB index, so it's safe to scatter now.
    /// unsafe { vals.scatter_select_unchecked(&mut vec, enable, idxs); }
    /// // The second write to index 0 was masked, thus omitted.
    /// assert_eq!(vec, vec![-41, 11, 12, 82, 14, 15, 16, 17, 18]);
    /// ```
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn scatter_select_unchecked(
        self,
        slice: &mut [T],
        enable: Mask<isize, N>,
        idxs: Simd<usize, N>,
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
            let base_ptr = Simd::<*mut T, N>::splat(slice.as_mut_ptr());
            // Ferris forgive me, I have done pointer arithmetic here.
            let ptrs = base_ptr.wrapping_add(idxs);
            // The ptrs have been bounds-masked to prevent memory-unsafe writes insha'allah
            self.scatter_select_ptr(ptrs, enable);
            // Cleared ☢️ *mut T Zone
        }
    }

    /// Write pointers elementwise into a SIMD vector.
    ///
    /// # Safety
    ///
    /// Each write must satisfy the same conditions as [`core::ptr::write`].
    ///
    /// # Example
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Simd, SimdMutPtr};
    /// let mut values = [0; 4];
    /// let offset = Simd::from_array([3, 2, 1, 0]);
    /// let ptrs = Simd::splat(values.as_mut_ptr()).wrapping_add(offset);
    /// unsafe { Simd::from_array([6, 3, 5, 7]).scatter_ptr(ptrs); }
    /// assert_eq!(values, [7, 5, 3, 6]);
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn scatter_ptr(self, dest: Simd<*mut T, N>) {
        // Safety: The caller is responsible for upholding all invariants
        unsafe { self.scatter_select_ptr(dest, Mask::splat(true)) }
    }

    /// Conditionally write pointers elementwise into a SIMD vector.
    /// The mask `enable`s all `true` pointers and disables all `false` pointers.
    /// If a pointer is disabled, the write to its pointee is skipped.
    ///
    /// # Safety
    ///
    /// Enabled pointers must satisfy the same conditions as [`core::ptr::write`].
    ///
    /// # Example
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::{Mask, Simd, SimdMutPtr};
    /// let mut values = [0; 4];
    /// let offset = Simd::from_array([3, 2, 1, 0]);
    /// let ptrs = Simd::splat(values.as_mut_ptr()).wrapping_add(offset);
    /// let enable = Mask::from_array([true, true, false, false]);
    /// unsafe { Simd::from_array([6, 3, 5, 7]).scatter_select_ptr(ptrs, enable); }
    /// assert_eq!(values, [0, 0, 3, 6]);
    /// ```
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn scatter_select_ptr(self, dest: Simd<*mut T, N>, enable: Mask<isize, N>) {
        // Safety: The caller is responsible for upholding all invariants
        unsafe { intrinsics::simd_scatter(self, dest, enable.to_int()) }
    }
}

impl<T, const N: usize> Copy for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
}

impl<T, const N: usize> Clone for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> Default for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement + Default,
{
    #[inline]
    fn default() -> Self {
        Self::splat(T::default())
    }
}

impl<T, const N: usize> PartialEq for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Safety: All SIMD vectors are SimdPartialEq, and the comparison produces a valid mask.
        let mask = unsafe {
            let tfvec: Simd<<T as SimdElement>::Mask, N> = intrinsics::simd_eq(*self, *other);
            Mask::from_int_unchecked(tfvec)
        };

        // Two vectors are equal if all elements are equal when compared elementwise
        mask.all()
    }

    #[allow(clippy::partialeq_ne_impl)]
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        // Safety: All SIMD vectors are SimdPartialEq, and the comparison produces a valid mask.
        let mask = unsafe {
            let tfvec: Simd<<T as SimdElement>::Mask, N> = intrinsics::simd_ne(*self, *other);
            Mask::from_int_unchecked(tfvec)
        };

        // Two vectors are non-equal if any elements are non-equal when compared elementwise
        mask.any()
    }
}

impl<T, const N: usize> PartialOrd for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement + PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        // TODO use SIMD equality
        self.to_array().partial_cmp(other.as_ref())
    }
}

impl<T, const N: usize> Eq for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement + Eq,
{
}

impl<T, const N: usize> Ord for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // TODO use SIMD equality
        self.to_array().cmp(other.as_ref())
    }
}

impl<T, const N: usize> core::hash::Hash for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
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
impl<T, const N: usize> AsRef<[T; N]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        self.as_array()
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        self.as_mut_array()
    }
}

// slice references
impl<T, const N: usize> AsRef<[T]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_array()
    }
}

impl<T, const N: usize> AsMut<[T]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_array()
    }
}

// vector/array conversion
impl<T, const N: usize> From<[T; N]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn from(array: [T; N]) -> Self {
        Self::from_array(array)
    }
}

impl<T, const N: usize> From<Simd<T, N>> for [T; N]
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    #[inline]
    fn from(vector: Simd<T, N>) -> Self {
        vector.to_array()
    }
}

impl<T, const N: usize> TryFrom<&[T]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    type Error = core::array::TryFromSliceError;

    #[inline]
    fn try_from(slice: &[T]) -> Result<Self, core::array::TryFromSliceError> {
        Ok(Self::from_array(slice.try_into()?))
    }
}

impl<T, const N: usize> TryFrom<&mut [T]> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
{
    type Error = core::array::TryFromSliceError;

    #[inline]
    fn try_from(slice: &mut [T]) -> Result<Self, core::array::TryFromSliceError> {
        Ok(Self::from_array(slice.try_into()?))
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

impl<T> Sealed for *const T {}

// Safety: (thin) const pointers are valid SIMD element types, and are supported by this API
//
// Fat pointers may be supported in the future.
unsafe impl<T> SimdElement for *const T
where
    T: core::ptr::Pointee<Metadata = ()>,
{
    type Mask = isize;
}

impl<T> Sealed for *mut T {}

// Safety: (thin) mut pointers are valid SIMD element types, and are supported by this API
//
// Fat pointers may be supported in the future.
unsafe impl<T> SimdElement for *mut T
where
    T: core::ptr::Pointee<Metadata = ()>,
{
    type Mask = isize;
}
