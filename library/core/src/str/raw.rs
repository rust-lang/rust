use crate::{
    intrinsics::{assert_unsafe_precondition, is_aligned_and_not_null, is_valid_allocation_size},
    ptr,
};

/// Forms a [`str`] from a pointer and a length.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `data` must be [valid] for reads for `len` bytes,
///   and must be properly aligned. This means in particular:
///
///     * The entire memory range of this string slice must be contained within a single allocated object!
///       Slices can never span across multiple allocated objects. See [below](#incorrect-usage)
///       for an example incorrectly not taking this into account.
///     * `data` must be non-null and aligned even for zero-length string slices. One
///       reason for this is that enum layout optimizations may rely on references
///       (including slices of any length) being aligned and non-null to distinguish
///       them from other data. You can obtain a pointer that is usable as `data`
///       for zero-length slices using [`NonNull::dangling()`].
///
/// * `data` must point to `len` consecutive properly initialized values of type `u8`.
///
/// * The memory referenced by the returned slice must not be mutated for the duration
///   of lifetime `'a`, except inside an `UnsafeCell`.
///
/// * The total size `len` of the string slice must be no larger than `isize::MAX`.
///   See the safety documentation of [`pointer::offset`].
///
/// * All pointed-to bytes are valid UTF-8
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
/// use std::str;
///
/// // manifest a str for a single byte
/// let x = b'a';
/// let ptr = &x as *const _;
/// let string = unsafe { str::from_raw_parts(ptr, 1) };
/// assert_eq!(string, "a");
/// ```
///
/// ### Incorrect usage
///
/// The following `join_strings` function is **unsound** ⚠️
///
/// ```rust,no_run
/// use std::str;
///
/// fn join_strings<'a, T>(fst: &'a str, snd: &'a str) -> &'a str {
///     let fst_end = fst.as_ptr().wrapping_add(fst.len());
///     let snd_start = snd.as_ptr();
///     assert_eq!(fst_end, snd_start, "Slices must be contiguous!");
///     unsafe {
///         // The assertion above ensures `fst` and `snd` are contiguous, but they might
///         // still be contained within _different allocated objects_, in which case
///         // creating this string slice is undefined behavior.
///         str::from_raw_parts(fst.as_ptr(), fst.len() + snd.len())
///     }
/// }
///
/// fn main() {
///     // `a` and `b` are different allocated objects...
///     let a = "foo";
///     let b = "bar";
///     // ... which may nevertheless be laid out contiguously in memory: | a | b |
///     let _ = join_slices(a, b); // UB
/// }
/// ```
///
/// [valid]: ptr#safety
/// [`NonNull::dangling()`]: ptr::NonNull::dangling
#[inline]
#[must_use]
#[unstable(feature = "str_from_raw_parts", issue = "none")]
pub const unsafe fn from_raw_parts<'a>(data: *const u8, len: usize) -> &'a str {
    // SAFETY: the caller must uphold the safety contract for `from_raw_parts`.
    unsafe {
        assert_unsafe_precondition!(
            "str::from_raw_parts requires the pointer to be aligned and non-null, \
            the total size of the str not to exceed `isize::MAX` and that the pointed-to bytes are valid utf8",
            [](data: *const u8, len: usize) => is_aligned_and_not_null(data)
                && is_valid_allocation_size::<u8>(len)
                && super::validations::run_utf8_validation(crate::slice::from_raw_parts(data, len)).is_ok()
        );

        &*ptr::from_raw_parts(data.cast(), len)
    }
}

/// Performs the same functionality as [`from_raw_parts`], except that a
/// mutable [`str`] is returned.
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `data` must be [valid] for both reads and writes for `len` bytes,
///   and it must be properly aligned. This means in particular:
///
///     * The entire memory range of this string slice must be contained within a single allocated object!
///       String slices can never span across multiple allocated objects.
///     * `data` must be non-null and aligned even for zero-length string slices. One
///       reason for this is that enum layout optimizations may rely on references
///       (including slices of any length) being aligned and non-null to distinguish
///       them from other data. You can obtain a pointer that is usable as `data`
///       for zero-length slices using [`NonNull::dangling()`].
///
/// * `data` must point to `len` consecutive properly initialized values of type `u8`.
///
/// * The memory referenced by the returned slice must not be accessed through any other pointer
///   (not derived from the return value) for the duration of lifetime `'a`.
///   Both read and write accesses are forbidden.
///
/// * The total size `len` of the string slice must be no larger than `isize::MAX`.
///   See the safety documentation of [`pointer::offset`].
///
/// * All pointed-to bytes are valid UTF-8
///
/// [valid]: ptr#safety
/// [`NonNull::dangling()`]: ptr::NonNull::dangling
#[inline]
#[must_use]
#[unstable(feature = "str_from_raw_parts", issue = "none")]
#[rustc_const_unstable(feature = "const_str_from_raw_parts_mut", issue = "none")]
pub const unsafe fn from_raw_parts_mut<'a>(data: *mut u8, len: usize) -> &'a mut str {
    // SAFETY: the caller must uphold the safety contract for `from_raw_parts_mut`.
    unsafe {
        assert_unsafe_precondition!(
            "str::from_raw_parts_mut requires the pointer to be aligned and non-null, \
            the total size of the str not to exceed `isize::MAX` and that the pointed-to bytes are valid utf8",
            [](data: *mut u8, len: usize) => is_aligned_and_not_null(data)
                && is_valid_allocation_size::<u8>(len)
                && super::validations::run_utf8_validation(crate::slice::from_raw_parts(data, len)).is_ok()
        );

        &mut *ptr::from_raw_parts_mut(data.cast(), len)
    }
}
