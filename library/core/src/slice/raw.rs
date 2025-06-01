//! Free functions to create `&[T]` and `&mut [T]`.

use crate::ops::Range;
use crate::{array, ptr, ub_checks};

/// Forms a slice from a pointer and a length.
///
/// The `len` argument is the number of **elements**, not the number of bytes.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `data` must be non-null, [valid] for reads for `len * size_of::<T>()` many bytes,
///   and it must be properly aligned. This means in particular:
///
///     * The entire memory range of this slice must be contained within a single allocation!
///       Slices can never span across multiple allocations. See [below](#incorrect-usage)
///       for an example incorrectly not taking this into account.
///     * `data` must be non-null and aligned even for zero-length slices or slices of ZSTs. One
///       reason for this is that enum layout optimizations may rely on references
///       (including slices of any length) being aligned and non-null to distinguish
///       them from other data. You can obtain a pointer that is usable as `data`
///       for zero-length slices using [`NonNull::dangling()`].
///
/// * `data` must point to `len` consecutive properly initialized values of type `T`.
///
/// * The memory referenced by the returned slice must not be mutated for the duration
///   of lifetime `'a`, except inside an `UnsafeCell`.
///
/// * The total size `len * size_of::<T>()` of the slice must be no larger than `isize::MAX`,
///   and adding that size to `data` must not "wrap around" the address space.
///   See the safety documentation of [`pointer::offset`].
///
/// # Caveat
///
/// The lifetime for the returned slice is inferred from its usage. To
/// prevent accidental misuse, it's suggested to tie the lifetime to whichever
/// source lifetime is safe in the context, such as by providing a helper
/// function taking the lifetime of a host value for the slice, or by explicit
/// annotation.
///
/// # Examples
///
/// ```
/// use std::slice;
///
/// // manifest a slice for a single element
/// let x = 42;
/// let ptr = &x as *const _;
/// let slice = unsafe { slice::from_raw_parts(ptr, 1) };
/// assert_eq!(slice[0], 42);
/// ```
///
/// ### Incorrect usage
///
/// The following `join_slices` function is **unsound** ⚠️
///
/// ```rust,no_run
/// use std::slice;
///
/// fn join_slices<'a, T>(fst: &'a [T], snd: &'a [T]) -> &'a [T] {
///     let fst_end = fst.as_ptr().wrapping_add(fst.len());
///     let snd_start = snd.as_ptr();
///     assert_eq!(fst_end, snd_start, "Slices must be contiguous!");
///     unsafe {
///         // The assertion above ensures `fst` and `snd` are contiguous, but they might
///         // still be contained within _different allocations_, in which case
///         // creating this slice is undefined behavior.
///         slice::from_raw_parts(fst.as_ptr(), fst.len() + snd.len())
///     }
/// }
///
/// fn main() {
///     // `a` and `b` are different allocations...
///     let a = 42;
///     let b = 27;
///     // ... which may nevertheless be laid out contiguously in memory: | a | b |
///     let _ = join_slices(slice::from_ref(&a), slice::from_ref(&b)); // UB
/// }
/// ```
///
/// ### FFI: Handling null pointers
///
/// In languages such as C++, pointers to empty collections are not guaranteed to be non-null.
/// When accepting such pointers, they have to be checked for null-ness to avoid undefined
/// behavior.
///
/// ```
/// use std::slice;
///
/// /// Sum the elements of an FFI slice.
/// ///
/// /// # Safety
/// ///
/// /// If ptr is not NULL, it must be correctly aligned and
/// /// point to `len` initialized items of type `f32`.
/// unsafe extern "C" fn sum_slice(ptr: *const f32, len: usize) -> f32 {
///     let data = if ptr.is_null() {
///         // `len` is assumed to be 0.
///         &[]
///     } else {
///         // SAFETY: see function docstring.
///         unsafe { slice::from_raw_parts(ptr, len) }
///     };
///     data.into_iter().sum()
/// }
///
/// // This could be the result of C++'s std::vector::data():
/// let ptr = std::ptr::null();
/// // And this could be std::vector::size():
/// let len = 0;
/// assert_eq!(unsafe { sum_slice(ptr, len) }, 0.0);
/// ```
///
/// [valid]: ptr#safety
/// [`NonNull::dangling()`]: ptr::NonNull::dangling
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_slice_from_raw_parts", since = "1.64.0")]
#[must_use]
#[rustc_diagnostic_item = "slice_from_raw_parts"]
#[track_caller]
pub const unsafe fn from_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T] {
    // SAFETY: the caller must uphold the safety contract for `from_raw_parts`.
    unsafe {
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "slice::from_raw_parts requires the pointer to be aligned and non-null, and the total size of the slice not to exceed `isize::MAX`",
            (
                data: *mut () = data as *mut (),
                size: usize = size_of::<T>(),
                align: usize = align_of::<T>(),
                len: usize = len,
            ) =>
            ub_checks::maybe_is_aligned_and_not_null(data, align, false)
                && ub_checks::is_valid_allocation_size(size, len)
        );
        &*ptr::slice_from_raw_parts(data, len)
    }
}

/// Performs the same functionality as [`from_raw_parts`], except that a
/// mutable slice is returned.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `data` must be non-null, [valid] for both reads and writes for `len * size_of::<T>()` many bytes,
///   and it must be properly aligned. This means in particular:
///
///     * The entire memory range of this slice must be contained within a single allocation!
///       Slices can never span across multiple allocations.
///     * `data` must be non-null and aligned even for zero-length slices or slices of ZSTs. One
///       reason for this is that enum layout optimizations may rely on references
///       (including slices of any length) being aligned and non-null to distinguish
///       them from other data. You can obtain a pointer that is usable as `data`
///       for zero-length slices using [`NonNull::dangling()`].
///
/// * `data` must point to `len` consecutive properly initialized values of type `T`.
///
/// * The memory referenced by the returned slice must not be accessed through any other pointer
///   (not derived from the return value) for the duration of lifetime `'a`.
///   Both read and write accesses are forbidden.
///
/// * The total size `len * size_of::<T>()` of the slice must be no larger than `isize::MAX`,
///   and adding that size to `data` must not "wrap around" the address space.
///   See the safety documentation of [`pointer::offset`].
///
/// [valid]: ptr#safety
/// [`NonNull::dangling()`]: ptr::NonNull::dangling
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_slice_from_raw_parts_mut", since = "1.83.0")]
#[must_use]
#[rustc_diagnostic_item = "slice_from_raw_parts_mut"]
#[track_caller]
pub const unsafe fn from_raw_parts_mut<'a, T>(data: *mut T, len: usize) -> &'a mut [T] {
    // SAFETY: the caller must uphold the safety contract for `from_raw_parts_mut`.
    unsafe {
        ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "slice::from_raw_parts_mut requires the pointer to be aligned and non-null, and the total size of the slice not to exceed `isize::MAX`",
            (
                data: *mut () = data as *mut (),
                size: usize = size_of::<T>(),
                align: usize = align_of::<T>(),
                len: usize = len,
            ) =>
            ub_checks::maybe_is_aligned_and_not_null(data, align, false)
                && ub_checks::is_valid_allocation_size(size, len)
        );
        &mut *ptr::slice_from_raw_parts_mut(data, len)
    }
}

/// Converts a reference to T into a slice of length 1 (without copying).
#[stable(feature = "from_ref", since = "1.28.0")]
#[rustc_const_stable(feature = "const_slice_from_ref_shared", since = "1.63.0")]
#[must_use]
pub const fn from_ref<T>(s: &T) -> &[T] {
    array::from_ref(s)
}

/// Converts a reference to T into a slice of length 1 (without copying).
#[stable(feature = "from_ref", since = "1.28.0")]
#[rustc_const_stable(feature = "const_slice_from_ref", since = "1.83.0")]
#[must_use]
pub const fn from_mut<T>(s: &mut T) -> &mut [T] {
    array::from_mut(s)
}

/// Forms a slice from a pointer range.
///
/// This function is useful for interacting with foreign interfaces which
/// use two pointers to refer to a range of elements in memory, as is
/// common in C++.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * The `start` pointer of the range must be a non-null, [valid] and properly aligned pointer
///   to the first element of a slice.
///
/// * The `end` pointer must be a [valid] and properly aligned pointer to *one past*
///   the last element, such that the offset from the end to the start pointer is
///   the length of the slice.
///
/// * The entire memory range of this slice must be contained within a single allocation!
///   Slices can never span across multiple allocations.
///
/// * The range must contain `N` consecutive properly initialized values of type `T`.
///
/// * The memory referenced by the returned slice must not be mutated for the duration
///   of lifetime `'a`, except inside an `UnsafeCell`.
///
/// * The total length of the range must be no larger than `isize::MAX`,
///   and adding that size to `start` must not "wrap around" the address space.
///   See the safety documentation of [`pointer::offset`].
///
/// Note that a range created from [`slice::as_ptr_range`] fulfills these requirements.
///
/// # Panics
///
/// This function panics if `T` is a Zero-Sized Type (“ZST”).
///
/// # Caveat
///
/// The lifetime for the returned slice is inferred from its usage. To
/// prevent accidental misuse, it's suggested to tie the lifetime to whichever
/// source lifetime is safe in the context, such as by providing a helper
/// function taking the lifetime of a host value for the slice, or by explicit
/// annotation.
///
/// # Examples
///
/// ```
/// #![feature(slice_from_ptr_range)]
///
/// use core::slice;
///
/// let x = [1, 2, 3];
/// let range = x.as_ptr_range();
///
/// unsafe {
///     assert_eq!(slice::from_ptr_range(range), &x);
/// }
/// ```
///
/// [valid]: ptr#safety
#[unstable(feature = "slice_from_ptr_range", issue = "89792")]
#[rustc_const_unstable(feature = "const_slice_from_ptr_range", issue = "89792")]
#[track_caller]
pub const unsafe fn from_ptr_range<'a, T>(range: Range<*const T>) -> &'a [T] {
    // SAFETY: the caller must uphold the safety contract for `from_ptr_range`.
    unsafe { from_raw_parts(range.start, range.end.offset_from_unsigned(range.start)) }
}

/// Forms a mutable slice from a pointer range.
///
/// This is the same functionality as [`from_ptr_range`], except that a
/// mutable slice is returned.
///
/// This function is useful for interacting with foreign interfaces which
/// use two pointers to refer to a range of elements in memory, as is
/// common in C++.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * The `start` pointer of the range must be a non-null, [valid] and properly aligned pointer
///   to the first element of a slice.
///
/// * The `end` pointer must be a [valid] and properly aligned pointer to *one past*
///   the last element, such that the offset from the end to the start pointer is
///   the length of the slice.
///
/// * The entire memory range of this slice must be contained within a single allocation!
///   Slices can never span across multiple allocations.
///
/// * The range must contain `N` consecutive properly initialized values of type `T`.
///
/// * The memory referenced by the returned slice must not be accessed through any other pointer
///   (not derived from the return value) for the duration of lifetime `'a`.
///   Both read and write accesses are forbidden.
///
/// * The total length of the range must be no larger than `isize::MAX`,
///   and adding that size to `start` must not "wrap around" the address space.
///   See the safety documentation of [`pointer::offset`].
///
/// Note that a range created from [`slice::as_mut_ptr_range`] fulfills these requirements.
///
/// # Panics
///
/// This function panics if `T` is a Zero-Sized Type (“ZST”).
///
/// # Caveat
///
/// The lifetime for the returned slice is inferred from its usage. To
/// prevent accidental misuse, it's suggested to tie the lifetime to whichever
/// source lifetime is safe in the context, such as by providing a helper
/// function taking the lifetime of a host value for the slice, or by explicit
/// annotation.
///
/// # Examples
///
/// ```
/// #![feature(slice_from_ptr_range)]
///
/// use core::slice;
///
/// let mut x = [1, 2, 3];
/// let range = x.as_mut_ptr_range();
///
/// unsafe {
///     assert_eq!(slice::from_mut_ptr_range(range), &mut [1, 2, 3]);
/// }
/// ```
///
/// [valid]: ptr#safety
#[unstable(feature = "slice_from_ptr_range", issue = "89792")]
#[rustc_const_unstable(feature = "const_slice_from_mut_ptr_range", issue = "89792")]
pub const unsafe fn from_mut_ptr_range<'a, T>(range: Range<*mut T>) -> &'a mut [T] {
    // SAFETY: the caller must uphold the safety contract for `from_mut_ptr_range`.
    unsafe { from_raw_parts_mut(range.start, range.end.offset_from_unsigned(range.start)) }
}
