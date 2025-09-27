use core::alloc::{AllocError, Allocator};
#[cfg(not(no_global_oom_handling))]
use core::mem;
#[cfg(not(no_global_oom_handling))]
use core::mem::DropGuard;
#[cfg(not(no_global_oom_handling))]
use core::ptr::{self, NonNull};

#[cfg(not(no_global_oom_handling))]
use crate::alloc;
use crate::raw_rc::RefCounts;
use crate::raw_rc::rc_layout::RcLayout;
use crate::raw_rc::rc_value_pointer::RcValuePointer;

/// Allocates uninitialized memory for a reference-counted allocation with allocator `alloc` and
/// layout `RcLayout`. Returns a pointer to the value location.
#[inline]
fn allocate_uninit_raw_bytes<A>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<RcValuePointer, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate(rc_layout.get());

    allocation_result.map(|allocation_ptr| {
        // SAFETY: `allocation_ptr` is allocated with `rc_layout`, so the safety requirement of
        // `RcValuePointer::from_allocation_ptr` is trivially satisfied.
        unsafe { RcValuePointer::from_allocation_ptr(allocation_ptr.cast(), rc_layout) }
    })
}

/// Allocates zeroed memory for a reference-counted allocation with allocator `alloc` and layout
/// `RcLayout`. Returns a pointer to the value location.
#[inline]
fn allocate_zeroed_raw_bytes<A>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<RcValuePointer, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate_zeroed(rc_layout.get());

    allocation_result.map(|allocation_ptr| {
        // SAFETY: `allocation_ptr` is allocated with `rc_layout`, so the safety requirement of
        // `RcValuePointer::from_allocation_ptr` is trivially satisfied.
        unsafe { RcValuePointer::from_allocation_ptr(allocation_ptr.cast(), rc_layout) }
    })
}

/// Initializes reference counters in a reference-counted allocation pointed to by `value_ptr`
/// with strong count of `STRONG_COUNT` and weak count of 1.
///
/// # Safety
///
/// - `value_ptr` points to a valid reference-counted allocation.
#[inline]
unsafe fn init_rc_allocation<const STRONG_COUNT: usize>(value_ptr: RcValuePointer) {
    // SAFETY: Caller guarantees the `value_ptr` points to a valid reference-counted allocation, so
    // we can write to the corresponding `RefCounts` object.
    unsafe { value_ptr.ref_counts_ptr().write(const { RefCounts::new(STRONG_COUNT) }) };
}

/// Tries to allocate a chunk of reference-counted memory that is described by `rc_layout` with
/// `alloc`. The allocated memory has strong count of `STRONG_COUNT` and weak count of 1.
pub(crate) fn try_allocate_uninit_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<RcValuePointer, AllocError>
where
    A: Allocator,
{
    let value_ptr = allocate_uninit_raw_bytes(alloc, rc_layout)?;

    // SAFETY: `value_ptr` is newly allocated, so it is guaranteed to be valid.
    unsafe { init_rc_allocation::<STRONG_COUNT>(value_ptr) };

    Ok(value_ptr)
}

/// Creates an allocator of type `A`, then tries to allocate a chunk of reference-counted memory
/// that is described by `rc_layout`.
pub(crate) fn try_allocate_uninit<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> Result<(RcValuePointer, A), AllocError>
where
    A: Allocator + Default,
{
    let alloc = A::default();

    try_allocate_uninit_in::<A, STRONG_COUNT>(&alloc, rc_layout).map(|value_ptr| (value_ptr, alloc))
}

/// Tries to allocate a reference-counted memory that is described by `rc_layout` with `alloc`. The
/// allocated memory has strong count of `STRONG_COUNT` and weak count of 1, and the value memory
/// is all zero bytes.
pub(crate) fn try_allocate_zeroed_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<RcValuePointer, AllocError>
where
    A: Allocator,
{
    let value_ptr = allocate_zeroed_raw_bytes(alloc, rc_layout)?;

    // SAFETY: `value_ptr` is newly allocated, so it is guaranteed to be valid.
    unsafe { init_rc_allocation::<STRONG_COUNT>(value_ptr) };

    Ok(value_ptr)
}

/// Creates an allocator of type `A`, then tries to allocate a chunk of reference-counted memory
/// with all zero bytes memory that is described by `rc_layout`.
pub(crate) fn try_allocate_zeroed<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> Result<(RcValuePointer, A), AllocError>
where
    A: Allocator + Default,
{
    let alloc = A::default();

    try_allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, rc_layout).map(|value_ptr| (value_ptr, alloc))
}

/// If `allocation_result` is `Ok`, initializes the reference counts with strong count
/// `STRONG_COUNT` and weak count of 1 and returns a pointer to the value object, otherwise panic
/// will be triggered by calling `alloc::handle_alloc_error`.
///
/// # Safety
///
/// If `allocation_result` is `Ok`, the pointer it contains must point to a valid reference-counted
/// allocation that is allocated with `rc_layout`.
#[cfg(not(no_global_oom_handling))]
#[inline]
unsafe fn handle_rc_allocation_result<const STRONG_COUNT: usize>(
    allocation_result: Result<RcValuePointer, AllocError>,
    rc_layout: RcLayout,
) -> RcValuePointer {
    match allocation_result {
        Ok(value_ptr) => {
            // SAFETY: Caller guarantees the `value_ptr` points to a valid reference-counted`
            // allocation.
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
pub(crate) fn allocate_uninit_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> RcValuePointer
where
    A: Allocator,
{
    let allocation_result = allocate_uninit_raw_bytes(alloc, rc_layout);

    // SAFETY: `allocation_result` is the allocation result using `rc_layout`, which satisfies the
    // safety requirement of `handle_rc_allocation_result`.
    unsafe { handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory that is
/// described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(crate) fn allocate_uninit<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> (RcValuePointer, A)
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
pub(crate) fn allocate_zeroed_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> RcValuePointer
where
    A: Allocator,
{
    let allocation_result = allocate_zeroed_raw_bytes(alloc, rc_layout);

    // SAFETY: `allocation_result` is the allocation result using `rc_layout`, which satisfies the
    // safety requirement of `handle_rc_allocation_result`.
    unsafe { handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory with all
/// zero bytes that is described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
pub(crate) fn allocate_zeroed<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> (RcValuePointer, A)
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
pub(crate) fn allocate_with_in<A, F, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
    f: F,
) -> RcValuePointer
where
    A: Allocator,
    F: FnOnce(RcValuePointer),
{
    /// # Safety
    ///
    /// - `value_ptr` points to a valid value location within a reference counted allocation
    ///   that can be described with `rc_layout` and can be deallocated with `alloc`.
    /// - No access to the allocation can happen if the destructor of the returned guard get called.
    unsafe fn deallocate_on_drop<'a, A>(
        value_ptr: RcValuePointer,
        alloc: &'a A,
        rc_layout: RcLayout,
    ) -> impl Drop + use<'a, A>
    where
        A: Allocator,
    {
        // SAFETY: Caller guarantees the validity of all arguments.
        DropGuard::new((), move |()| unsafe {
            deallocate::<A>(value_ptr, alloc, rc_layout);
        })
    }

    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(alloc, rc_layout);
    let guard = unsafe { deallocate_on_drop(value_ptr, alloc, rc_layout) };

    f(value_ptr);

    mem::forget(guard);

    value_ptr
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory that is
/// described by `rc_layout`. `f` will be called with a pointer that points the value storage to
/// initialize the allocated memory. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(crate) fn allocate_with<A, F, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
    f: F,
) -> (RcValuePointer, A)
where
    A: Allocator + Default,
    F: FnOnce(RcValuePointer),
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
pub(crate) unsafe fn allocate_with_bytes_in<A, const STRONG_COUNT: usize>(
    src_ptr: NonNull<()>,
    alloc: &A,
    rc_layout: RcLayout,
) -> RcValuePointer
where
    A: Allocator,
{
    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(alloc, rc_layout);
    let value_size = rc_layout.value_size();

    unsafe {
        ptr::copy_nonoverlapping::<u8>(
            src_ptr.as_ptr().cast(),
            value_ptr.as_ptr().as_ptr().cast(),
            value_size,
        );
    }

    value_ptr
}

/// Allocates a chunk of reference-counted memory with a value that is copied from `value`. This is
/// safe because the return value is a pointer, which will not cause double unless caller calls the
/// destructor manually, which requires `unsafe` codes.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(crate) fn allocate_with_value_in<T, A, const STRONG_COUNT: usize>(
    src: &T,
    alloc: &A,
) -> NonNull<T>
where
    T: ?Sized,
    A: Allocator,
{
    let src_ptr = NonNull::from(src);

    // SAFETY: `src_ptr` is created from a reference, so it has correct metadata.
    let rc_layout = unsafe { RcLayout::from_value_ptr(src_ptr) };

    let (src_ptr, metadata) = src_ptr.to_raw_parts();

    // SAFETY: `src_ptr` comes from a reference to `T`, so it is guaranteed to have enough data to
    // fill the value in an allocation that is described by `rc_layout`.
    let value_ptr = unsafe { allocate_with_bytes_in::<A, STRONG_COUNT>(src_ptr, alloc, rc_layout) };

    NonNull::from_raw_parts(value_ptr.as_ptr(), metadata)
}

/// Creates an allocator of type `A`, then allocates a chunk of reference-counted memory with value
/// copied from `value`.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(crate) fn allocate_with_value<T, A, const STRONG_COUNT: usize>(value: &T) -> (NonNull<T>, A)
where
    T: ?Sized,
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_with_value_in::<T, A, STRONG_COUNT>(value, &alloc);

    (value_ptr, alloc)
}

/// Deallocates a reference-counted allocation with a value object pointed to by `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` points to a valid reference-counted allocation that is allocated using
///   `rc_layout`.
#[inline]
pub(crate) unsafe fn deallocate<A>(value_ptr: RcValuePointer, alloc: &A, rc_layout: RcLayout)
where
    A: Allocator,
{
    let value_offset = rc_layout.value_offset();
    let allocation_ptr = unsafe { value_ptr.as_ptr().byte_sub(value_offset) };

    unsafe { alloc.deallocate(allocation_ptr.cast(), rc_layout.get()) }
}
