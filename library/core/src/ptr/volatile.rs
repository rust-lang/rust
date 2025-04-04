use crate::mem::SizedTypeProperties;
use crate::intrinsics;
use crate::macros::cfg;
use crate::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, Atomicu64};

/// Performs a volatile read of the value from `src` without moving it. This
/// leaves the memory in `src` unchanged.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `read_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `src` must be [valid] for reads.
///
/// * `src` must be properly aligned.
///
/// * `src` must point to a properly initialized value of type `T`.
///
/// Like [read], `read_volatile` creates a bitwise copy of `T`, regardless of
/// whether `T` is [`Copy`]. If `T` is not [`Copy`], using both the returned
/// value and the value at `*src` can [violate memory safety][read-ownership].
/// However, storing non-[`Copy`] types in volatile memory is almost certainly
/// incorrect.
///
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
///
/// [valid]: crate::ptr#safety
/// [read-ownership]: crate::ptr::read#ownership-of-the-returned-value
/// [read]: crate::ptr::read
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `read_volatile` and any write operation to the same location
/// is undefined behavior.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let x = 12;
/// let y = &x as *const i32;
///
/// unsafe {
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
#[cfg_attr(miri, track_caller)] // Even without panics, this helps for Miri backtraces
#[rustc_diagnostic_item = "ptr_read_volatile"]
pub unsafe fn read_volatile<T>(src: *const T) -> T {
    // SAFETY: the caller must uphold the safety contract for `volatile_load`.
    unsafe {
        crate::ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::read_volatile requires that the pointer argument is aligned and non-null",
            (
                addr: *const () = src as *const (),
                align: usize = align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => crate::ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
        );
        match size_of<T>() {
            1 => if cfg!(target_has_atomic_load_store = "8") && align_of::<T>() == align_of::<AtomicU8>() {
                    intrinsics::atomic_load_relaxed(src)
                } else {
                    intrinsics::volatile_load(dst, val)
                }
            2 => if cfg!(target_has_atomic_load_store = "16") && align_of::<T>() == align_of::<AtomicU16>() {
                    intrinsics::atomic_load_relaxed(src)
                } else {
                    intrinsics::volatile_load(dst, val)
                }
            4 => if cfg!(target_has_atomic_load_store = "32") && align_of::<T>() == align_of::<AtomicU32>() {
                    intrinsics::atomic_load_relaxed(src)
                } else {
                    intrinsics::volatile_load(dst, val)
                }
            8 => if cfg!(target_has_atomic_load_store = "64") && align_of::<T>() == align_of::<AtomicU64>() {
                    intrinsics::atomic_load_relaxed(src)
                } else {
                    intrinsics::volatile_load(dst, val)
                }
            _ => intrinsics::volatile_load(dst, val)
        }
    }
}

/// Performs a volatile write of a memory location with the given value without
/// reading or dropping the old value.
///
/// Volatile operations are intended to act on I/O memory, and are guaranteed
/// to not be elided or reordered by the compiler across other volatile
/// operations.
///
/// `write_volatile` does not drop the contents of `dst`. This is safe, but it
/// could leak allocations or resources, so care should be taken not to overwrite
/// an object that should be dropped.
///
/// Additionally, it does not drop `src`. Semantically, `src` is moved into the
/// location pointed to by `dst`.
///
/// # Notes
///
/// Rust does not currently have a rigorously and formally defined memory model,
/// so the precise semantics of what "volatile" means here is subject to change
/// over time. That being said, the semantics will almost always end up pretty
/// similar to [C11's definition of volatile][c11].
///
/// The compiler shouldn't change the relative order or number of volatile
/// memory operations. However, volatile memory operations on zero-sized types
/// (e.g., if a zero-sized type is passed to `write_volatile`) are noops
/// and may be ignored.
///
/// [c11]: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
///
/// # Safety
///
/// Behavior is undefined if any of the following conditions are violated:
///
/// * `dst` must be [valid] for writes.
///
/// * `dst` must be properly aligned.
///
/// Note that even if `T` has size `0`, the pointer must be properly aligned.
///
/// [valid]: crate::ptr#safety
///
/// Just like in C, whether an operation is volatile has no bearing whatsoever
/// on questions involving concurrent access from multiple threads. Volatile
/// accesses behave exactly like non-atomic accesses in that regard. In particular,
/// a race between a `write_volatile` and any other operation (reading or writing)
/// on the same location is undefined behavior.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let mut x = 0;
/// let y = &mut x as *mut i32;
/// let z = 12;
///
/// unsafe {
///     std::ptr::write_volatile(y, z);
///     assert_eq!(std::ptr::read_volatile(y), 12);
/// }
/// ```
#[inline]
#[stable(feature = "volatile", since = "1.9.0")]
#[cfg_attr(miri, track_caller)] // Even without panics, this helps for Miri backtraces
#[rustc_diagnostic_item = "ptr_write_volatile"]
pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
    // SAFETY: the caller must uphold the safety contract for `volatile_write`.
    unsafe {
        crate::ub_checks::assert_unsafe_precondition!(
            check_language_ub,
            "ptr::write_volatile requires that the pointer argument is aligned and non-null",
            (
                addr: *mut () = dst as *mut (),
                align: usize = crate::mem::align_of::<T>(),
                is_zst: bool = T::IS_ZST,
            ) => crate::ub_checks::maybe_is_aligned_and_not_null(addr, align, is_zst)
        );
        match size_of<T>() {
            1 => if cfg!(target_has_atomic_load_store = "8") && align_of::<T>() == align_of::<AtomicU8>() {
                    intrinsics::atomic_store_relaxed(dst, val)
                } else {
                    intrinsics::volatile_store(dst, val)
                }
            2 => if cfg!(target_has_atomic_load_store = "16") && align_of::<T>() == align_of::<AtomicU16>() {
                    intrinsics::atomic_store_relaxed(dst, val)
                } else {
                    intrinsics::volatile_store(dst, val)
                }
            4 => if cfg!(target_has_atomic_load_store = "32") && align_of::<T>() == align_of::<AtomicU32>() {
                    intrinsics::atomic_store_relaxed(dst, val)
                } else {
                    intrinsics::volatile_store(dst, val)
                }
            8 => if cfg!(target_has_atomic_load_store = "64") && align_of::<T>() == align_of::<AtomicU64>() {
                    intrinsics::atomic_store_relaxed(dst, val)
                } else {
                    intrinsics::volatile_store(dst, val)
                }
            _ => intrinsics::volatile_store(dst, val)
        }
    }
}
