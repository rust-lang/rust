//! Base implementation for `rc::{Rc, UniqueRc, Weak}` and `sync::{Arc, UniqueArc, Weak}`.
//!
//! # Allocation Memory Layout
//!
//! The memory layout of a reference-counted allocation is designed so that the memory that stores
//! the reference counts has a fixed offset to the memory that stores the value. In this way,
//! operations that only rely on reference counts can ignore the actual type of the contained value
//! and only care about the address of the contained value, which allows us to share code between
//! reference-counting pointers that have different types of contained values. This can potentially
//! reduce the binary size.
//!
//! Assume the type of the stored value is `T`, the allocation memory layout is designed as follows:
//!
//! - We use a `RefCounts` type to store the reference counts.
//! - The alignment of the allocation is `align_of::<RefCounts>().max(align_of::<T>())`.
//! - The value is stored at offset `size_of::<RefCounts>().next_multiple_of(align_of::<T>())`.
//! - The size of the allocation is
//!   `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) + size_of::<T>()`.
//! - The `RefCounts` object is stored at offset
//!   `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) - size_of::<RefCounts>()`.
//!
//! Here is a table showing the order and size of each component in an reference counted allocation
//! of a `T` value:
//!
//! | Component   | Size                                                                                |
//! | ----------- | ----------------------------------------------------------------------------------- |
//! | Padding     | `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) - size_of::<RefCounts>()` |
//! | `RefCounts` | `size_of::<RefCounts>()`                                                            |
//! | `T`         | `size_of::<T>()`                                                                    |
//!
//! This works because:
//!
//! - Both `RefCounts` and the object is stored in the allocation without overlapping.
//! - The `RefCounts` object is stored at offset
//!   `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) - size_of::<RefCounts>()`, which
//!   has a valid alignment for `RefCounts` because:
//!   - If `align_of::<T>() <= align_of::<RefCounts>()`, we have the offset being 0, which has a
//!     valid alignment for `RefCounts`.
//!   - If `align_of::<T>() > align_of::<RefCounts>()`, we have `align_of::<T>()` being a multiple
//!     of `align_of::<RefCounts>()`, since `size_of::<RefCounts>()` is also a multiple of
//!    `align_of::<RefCounts>()`, we conclude the offset also has a valid alignment for `RefCounts`.
//! - The value is stored at offset `size_of::<RefCounts>().next_multiple_of(align_of::<T>())`,
//!   which trivially satisfies the alignment requirement of `T`.
//! - The distance between the `RefCounts` object and the value is `size_of::<RefCounts>()`, a fixed
//!   value.
//!
//! So both the `RefCounts` object and the value object have their alignment and size requirements
//! satisfied. And we get a fixed offset between those two objects.
//!
//! # Reference-counting Pointer Design
//!
//! Both strong and weak reference-counting pointers store a pointer that points to the value
//! object in a reference-counted allocation, instead of a pointer to the beginning of the
//! allocation. This is based on the assumption that users access the contained value more
//! frequently than the reference counters. Also, this possibly allows us to enable some
//! optimizations like:
//!
//! - Making reference-counting pointers have ABI-compatible representation as raw pointers so we
//!   can use them directly in FFI interfaces.
//! - Converting `Option<Rc<T>>` to `Option<&T>` without checking for `None` values.
//! - Converting `&[Rc<T>]` to `&[&T]` with zero cost.

use core::alloc::{AllocError, Allocator};
use core::cell::UnsafeCell;
use core::ptr::NonNull;
#[cfg(not(no_global_oom_handling))]
use core::{mem, ptr};

#[cfg(not(no_global_oom_handling))]
use crate::alloc;
pub(crate) use crate::raw_rc::raw_rc::RawRc;
pub(crate) use crate::raw_rc::raw_unique_rc::RawUniqueRc;
pub(crate) use crate::raw_rc::raw_weak::RawWeak;
use crate::raw_rc::rc_layout::RcLayout;

mod raw_rc;
mod raw_unique_rc;
mod raw_weak;
mod rc_layout;

/// The return value type for `RcOps::make_mut`.
#[cfg(not(no_global_oom_handling))]
pub(crate) enum MakeMutStrategy {
    /// This `RawRc` is the only strong pointer that references the value, but there are weak
    /// pointers also referencing the value. Before returning, the strong reference count has been
    /// set to zero to prevent new strong pointers from being created through upgrading from weak
    /// pointers.
    Move,
    /// There is more than one strong pointer that references the value.
    Clone,
}

/// A trait for `rc` and `sync` modules to define their own implementation of reference-counting
/// behaviors.
///
/// # Safety
///
/// Implementors should implement each method according to its description.
pub(crate) unsafe trait RcOps: Sized {
    /// Increments a reference counter managed by `RawRc` and `RawWeak`. Currently, both strong and
    /// weak reference counters are incremented by this method.
    ///
    /// # Safety
    ///
    /// - `count` should only be handled by the same `RcOps` implementation.
    /// - The value of `count` should be non-zero.
    unsafe fn increment_ref_count(count: &UnsafeCell<usize>);

    /// Decrements a reference counter managed by `RawRc` and `RawWeak`. Currently, both strong and
    /// weak reference counters are decremented by this method. Returns whether the reference count
    /// becomes zero after decrementing.
    ///
    /// # Safety
    ///
    /// - `count` should only be handled by the same `RcOps` implementation.
    /// - The value of `count` should be non-zero.
    unsafe fn decrement_ref_count(count: &UnsafeCell<usize>) -> bool;

    /// Increments `strong_count` if and only if `strong_count` is non-zero. Returns whether
    /// incrementing is performed.
    ///
    /// # Safety
    ///
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    /// - `strong_count` should be provided by a `RawWeak` object.
    unsafe fn upgrade(strong_count: &UnsafeCell<usize>) -> bool;

    /// Increments `weak_count`. This is required instead of `increment_ref_count` because `Arc`
    /// requires additional synchronization with `is_unique`.
    ///
    /// # Safety
    ///
    /// - `weak_count` should only be handled by the same `RcOps` implementation.
    /// - `weak_count` should be provided by a `RawRc` object.
    unsafe fn downgrade(weak_count: &UnsafeCell<usize>);

    /// Decrements `strong_count` if and only if `strong_count` is 1. Returns true if decrementing
    /// is performed.
    ///
    /// # Safety
    ///
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    /// - `strong_count` should be provided by a `RawRc` object.
    unsafe fn lock_strong_count(strong_count: &UnsafeCell<usize>) -> bool;

    /// Sets `strong_count` to 1.
    ///
    /// # Safety
    ///
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    /// - `strong_count` should be provided by a `RawUniqueRc` object.
    unsafe fn unlock_strong_count(strong_count: &UnsafeCell<usize>);

    /// Returns whether both `strong_count` are 1 and `weak_count` is 1. Used by `RawRc::get_mut`.
    ///
    /// # Safety
    ///
    /// - `ref_counts` should only be handled by the same `RcOps` implementation.
    /// - `ref_counts` should be provided by a `RawRc` object.
    unsafe fn is_unique(ref_counts: &RefCounts) -> bool;

    /// Determines how to make a mutable reference from a `RawRc`:
    ///
    /// - If both strong count and weak count are 1, returns `None`.
    /// - If strong count is 1 and weak count is greater than 1, returns
    ///   `Some(MakeMutStrategy::Move)`.
    /// - If strong count is greater than 1, returns `Some(MakeMutStrategy::Clone)`.
    ///
    /// # Safety
    ///
    /// - `ref_counts` should only be handled by the same `RcOps` implementation.
    /// - `ref_counts` should be provided by a `RawRc` object.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn make_mut(ref_counts: &RefCounts) -> Option<MakeMutStrategy>;
}

/// Defines the `RefCounts` struct to store reference counts. The reference counters have suitable
/// alignment for atomic operations.
macro_rules! define_ref_counts {
    ($($target_pointer_width:literal => $align:literal,)*) => {
        $(
            /// Stores reference counts.
            #[cfg(target_pointer_width = $target_pointer_width)]
            #[repr(C, align($align))]
            pub(crate) struct RefCounts {
                /// Weak reference count (plus one if there are non-zero strong reference counts).
                pub(crate) weak: UnsafeCell<usize>,
                /// Strong reference count.
                pub(crate) strong: UnsafeCell<usize>,
            }
        )*
    };
}

// This ensures reference counters have correct alignment so that they can be treated as atomic
// reference counters for `Arc`.
define_ref_counts! {
    "16" => 2,
    "32" => 4,
    "64" => 8,
}

impl RefCounts {
    /// Creates a `RefCounts` with weak count of `1` and strong count of `strong_count`.
    const fn new(strong_count: usize) -> Self {
        Self { weak: UnsafeCell::new(1), strong: UnsafeCell::new(strong_count) }
    }
}

/// Gets a pointer to the `RefCounts` object in the same allocation with a value pointed to by
/// `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to a value object (can be uninitialized or dropped) that lives in a
///   reference-counted allocation.
#[inline]
unsafe fn ref_counts_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<RefCounts> {
    // SAFETY: Caller guarantees `value_ptr` point to the value inside some reference counted, our
    // implementation guarantees the `RefCounts` object has offset of `size_of::<RefCounts>()` to
    // the value, so we apply this offset.
    unsafe { value_ptr.cast::<RefCounts>().sub(1) }
}

/// Gets a pointer to the strong counter object in the same allocation with a value pointed to by
/// `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to a value object (can be uninitialized or dropped) that lives in a
///   reference-counted allocation.
#[inline]
unsafe fn strong_count_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<UnsafeCell<usize>> {
    let ref_counts_ptr = unsafe { ref_counts_ptr_from_value_ptr(value_ptr) };

    unsafe { NonNull::new_unchecked(&raw mut (*ref_counts_ptr.as_ptr()).strong) }
}

/// Gets a pointer to the weak counter object in the same allocation with a value pointed to by
/// `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to a value object (can be uninitialized or dropped) that lives in a
///   reference-counted allocation.
#[inline]
unsafe fn weak_count_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<UnsafeCell<usize>> {
    let ref_counts_ptr = unsafe { ref_counts_ptr_from_value_ptr(value_ptr) };

    unsafe { NonNull::new_unchecked(&raw mut (*ref_counts_ptr.as_ptr()).weak) }
}

/// Allocates uninitialized memory for a reference-counted allocation with allocator `alloc` and
/// layout `RcLayout`. Returns a pointer to the value location.
#[inline]
fn allocate_uninit_raw_bytes<A>(alloc: &A, rc_layout: RcLayout) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate(rc_layout.get());

    allocation_result
        .map(|allocation_ptr| unsafe { allocation_ptr.cast().byte_add(rc_layout.value_offset()) })
}

/// Allocates zeroed memory for a reference-counted allocation with allocator `alloc` and layout
/// `RcLayout`. Returns a pointer to the value location.
#[inline]
fn allocate_zeroed_raw_bytes<A>(alloc: &A, rc_layout: RcLayout) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate_zeroed(rc_layout.get());

    allocation_result
        .map(|allocation_ptr| unsafe { allocation_ptr.cast().byte_add(rc_layout.value_offset()) })
}

/// Initializes reference counters in a reference-counted allocation pointed to by `value_ptr`
/// with strong count of `STRONG_COUNT` and weak count of 1.
///
/// # Safety
///
/// - `value_ptr` points to a valid reference-counted allocation.
#[inline]
unsafe fn init_rc_allocation<const STRONG_COUNT: usize>(value_ptr: NonNull<()>) {
    unsafe {
        ref_counts_ptr_from_value_ptr(value_ptr).write(const { RefCounts::new(STRONG_COUNT) });
    }
}

/// Tries to allocate a chunk of reference-counted memory that is described by `rc_layout` with
/// `alloc`. The allocated memory has strong count of `STRONG_COUNT` and weak count of 1.
fn try_allocate_uninit_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    allocate_uninit_raw_bytes(alloc, rc_layout)
        .inspect(|&value_ptr| unsafe { init_rc_allocation::<STRONG_COUNT>(value_ptr) })
}

/// Creates an allocator of type `A`, then tries to allocate a chunk of reference-counted memory
/// that is described by `rc_layout`.
fn try_allocate_uninit<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> Result<(NonNull<()>, A), AllocError>
where
    A: Allocator + Default,
{
    let alloc = A::default();

    try_allocate_uninit_in::<A, STRONG_COUNT>(&alloc, rc_layout).map(|value_ptr| (value_ptr, alloc))
}

/// Tries to allocate a reference-counted memory that is described by `rc_layout` with `alloc`. The
/// allocated memory has strong count of `STRONG_COUNT` and weak count of 1, and the value memory
/// is all zero bytes.
fn try_allocate_zeroed_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    allocate_zeroed_raw_bytes(alloc, rc_layout)
        .inspect(|&value_ptr| unsafe { init_rc_allocation::<STRONG_COUNT>(value_ptr) })
}

/// Creates an allocator of type `A`, then tries to allocate a chunk of reference-counted memory
/// with all zero bytes memory that is described by `rc_layout`.
fn try_allocate_zeroed<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> Result<(NonNull<()>, A), AllocError>
where
    A: Allocator + Default,
{
    let alloc = A::default();

    try_allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, rc_layout).map(|value_ptr| (value_ptr, alloc))
}

/// If `allocation_result` is `Ok`, initializes the reference counts with strong count
/// `STRONG_COUNT` and weak count of 1 and returns a pointer to the value object, otherwise panic
/// will be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
unsafe fn handle_rc_allocation_result<const STRONG_COUNT: usize>(
    allocation_result: Result<NonNull<()>, AllocError>,
    rc_layout: RcLayout,
) -> NonNull<()> {
    match allocation_result {
        Ok(value_ptr) => {
            unsafe { init_rc_allocation::<STRONG_COUNT>(value_ptr) };

            value_ptr
        }
        Err(AllocError) => alloc::handle_alloc_error(rc_layout.get()),
    }
}

/// Allocates reference-counted memory that is described by `rc_layout` with `alloc`. The allocated
/// memory has strong count of `STRONG_COUNT` and weak count of 1. If the allocation fails, panic
/// will be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_uninit_in<A, const STRONG_COUNT: usize>(alloc: &A, rc_layout: RcLayout) -> NonNull<()>
where
    A: Allocator,
{
    let allocation_result = allocate_uninit_raw_bytes(alloc, rc_layout);

    unsafe { handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory that is
/// described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_uninit<A, const STRONG_COUNT: usize>(rc_layout: RcLayout) -> (NonNull<()>, A)
where
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(&alloc, rc_layout);

    (value_ptr, alloc)
}

/// Allocates reference-counted memory that is described by `rc_layout` with `alloc`. The allocated
/// memory has strong count of `STRONG_COUNT` and weak count of 1, and the value memory is all zero
/// bytes. If the allocation fails, panic will be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
fn allocate_zeroed_in<A, const STRONG_COUNT: usize>(alloc: &A, rc_layout: RcLayout) -> NonNull<()>
where
    A: Allocator,
{
    let allocation_result = allocate_zeroed_raw_bytes(alloc, rc_layout);

    unsafe { handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory with all
/// zero bytes that is described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
fn allocate_zeroed<A, const STRONG_COUNT: usize>(rc_layout: RcLayout) -> (NonNull<()>, A)
where
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, rc_layout);

    (value_ptr, alloc)
}

/// Allocates a reference-counted memory chunk for storing a value according to `rc_layout`, then
/// initialize the value with `f`. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_with_in<A, F, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
    f: F,
) -> NonNull<()>
where
    A: Allocator,
    F: FnOnce(NonNull<()>),
{
    struct Guard<'a, A>
    where
        A: Allocator,
    {
        value_ptr: NonNull<()>,
        alloc: &'a A,
        rc_layout: RcLayout,
    }

    impl<'a, A> Drop for Guard<'a, A>
    where
        A: Allocator,
    {
        fn drop(&mut self) {
            unsafe { deallocate::<A>(self.value_ptr, self.alloc, self.rc_layout) };
        }
    }

    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(alloc, rc_layout);
    let guard = Guard { value_ptr, alloc, rc_layout };

    f(value_ptr);

    mem::forget(guard);

    value_ptr
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory that is
/// described by `rc_layout`. `f` will be called with a pointer that points the value storage to
/// initialize the allocated memory. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_with<A, F, const STRONG_COUNT: usize>(rc_layout: RcLayout, f: F) -> (NonNull<()>, A)
where
    A: Allocator + Default,
    F: FnOnce(NonNull<()>),
{
    let alloc = A::default();
    let value_ptr = allocate_with_in::<A, F, STRONG_COUNT>(&alloc, rc_layout, f);

    (value_ptr, alloc)
}

/// Allocates reference-counted memory that has strong count of `STRONG_COUNT` and weak count of 1.
/// The value will be initialized with data pointed to by `src_ptr`.
///
/// # Safety
///
/// - Memory pointed to by `src_ptr` has enough data to read for filling the value in an allocation
///   that is described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
#[inline]
unsafe fn allocate_with_bytes_in<A, const STRONG_COUNT: usize>(
    src_ptr: NonNull<()>,
    alloc: &A,
    rc_layout: RcLayout,
) -> NonNull<()>
where
    A: Allocator,
{
    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(alloc, rc_layout);
    let value_size = rc_layout.value_size();

    unsafe {
        ptr::copy_nonoverlapping::<u8>(
            src_ptr.as_ptr().cast(),
            value_ptr.as_ptr().cast(),
            value_size,
        );
    }

    value_ptr
}

/// Allocates a chunk of reference-counted memory with a value that is copied from `value`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_with_value_in<T, A, const STRONG_COUNT: usize>(src: &T, alloc: &A) -> NonNull<T>
where
    T: ?Sized,
    A: Allocator,
{
    let src_ptr = NonNull::from(src);
    let rc_layout = unsafe { RcLayout::from_value_ptr(src_ptr) };
    let (src_ptr, metadata) = src_ptr.to_raw_parts();
    let value_ptr = unsafe { allocate_with_bytes_in::<A, STRONG_COUNT>(src_ptr, alloc, rc_layout) };

    NonNull::from_raw_parts(value_ptr, metadata)
}

/// Creates an allocator of type `A`, then allocates a chunk of reference-counted memory with value
/// copied from `value`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_with_value<T, A, const STRONG_COUNT: usize>(value: &T) -> (NonNull<T>, A)
where
    T: ?Sized,
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_with_value_in::<T, A, STRONG_COUNT>(value, &alloc);

    (value_ptr, alloc)
}

/// Deallocates a reference-counted allocation with a value object pointed to by `value_ptr`.
#[inline]
unsafe fn deallocate<A>(value_ptr: NonNull<()>, alloc: &A, rc_layout: RcLayout)
where
    A: Allocator,
{
    let value_offset = rc_layout.value_offset();
    let allocation_ptr = unsafe { value_ptr.byte_sub(value_offset) };

    unsafe { alloc.deallocate(allocation_ptr.cast(), rc_layout.get()) }
}
