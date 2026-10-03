use core::alloc::{AllocError, Allocator, Layout};
use core::clone::CloneToUninit;
#[cfg(not(no_global_oom_handling))]
use core::iter::TrustedLen;
use core::mem::DropGuard;
#[cfg(not(no_global_oom_handling))]
use core::mem::ManuallyDrop;
use core::ptr;
#[cfg(not(no_global_oom_handling))]
use core::ptr::NonNull;

#[cfg(not(no_global_oom_handling))]
use crate::alloc::{self, Global};
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
use crate::rcs::common::rc_layout::RcLayout;
use crate::rcs::common::rc_value_pointer::{ErasedRcValuePointer, RcValuePointer};
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

/// Tries to allocate uninitialized memory for a reference-counted allocation using allocator
/// `alloc` and layout `rc_layout`. Returns a pointer to the value location.
#[inline]
pub(in crate::rcs) fn try_allocate_uninit_in<H, A>(
    alloc: &A,
    rc_layout: RcLayout<H>,
) -> Result<ErasedRcValuePointer<H>, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate(rc_layout.get());

    allocation_result.map(|allocation_ptr| {
        // SAFETY: `allocation_ptr` is allocated with `rc_layout`, so the safety requirement of
        // `RcValuePointer::from_allocation_ptr` is trivially satisfied.
        unsafe {
            ErasedRcValuePointer::from_allocation_ptr(
                allocation_ptr.cast(),
                rc_layout.get().align(),
            )
        }
    })
}

/// Allocates uninitialized memory for a reference-counted allocation using allocator `alloc` and
/// layout `rc_layout`. Returns a pointer to the value location. If the allocation fails, panic will
/// be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_uninit_in<H, A>(
    alloc: &A,
    rc_layout: RcLayout<H>,
) -> ErasedRcValuePointer<H>
where
    A: Allocator,
{
    match try_allocate_uninit_in(alloc, rc_layout) {
        Ok(result) => result,
        Err(_) => alloc::handle_alloc_error(rc_layout.get()),
    }
}

/// Tries to allocate zeroed memory for a reference-counted allocation using allocator
/// `alloc` and layout `rc_layout`. Returns a pointer to the value location.
#[inline]
pub(in crate::rcs) fn try_allocate_zeroed_in<H, A>(
    alloc: &A,
    rc_layout: RcLayout<H>,
) -> Result<ErasedRcValuePointer<H>, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate_zeroed(rc_layout.get());

    allocation_result.map(|allocation_ptr| {
        // SAFETY: `allocation_ptr` is allocated with `rc_layout`, so the safety requirement of
        // `RcValuePointer::from_allocation_ptr` is trivially satisfied.
        unsafe {
            ErasedRcValuePointer::from_allocation_ptr(
                allocation_ptr.cast(),
                rc_layout.get().align(),
            )
        }
    })
}

/// Allocates zeroed memory for a reference-counted allocation using allocator `alloc` and
/// layout `rc_layout`. Returns a pointer to the value location. If the allocation fails, panic will
/// be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_zeroed_in<H, A>(
    alloc: &A,
    rc_layout: RcLayout<H>,
) -> ErasedRcValuePointer<H>
where
    A: Allocator,
{
    match try_allocate_zeroed_in(alloc, rc_layout) {
        Ok(result) => result,
        Err(_) => alloc::handle_alloc_error(rc_layout.get()),
    }
}

/// Deallocates a reference-counted allocation with a value object pointed to by `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` points to a valid reference-counted allocation that was allocated using
///   `rc_layout`.
#[inline]
pub(crate) unsafe fn deallocate<H, A>(
    alloc: &A,
    value_ptr: ErasedRcValuePointer<H>,
    rc_layout: RcLayout<H>,
) where
    A: Allocator,
{
    let value_offset = rc_layout.value_offset();

    // SAFETY: Caller guarantees the validity of `value_ptr`.
    unsafe {
        let allocation_ptr = value_ptr.ptr.byte_sub(value_offset);

        alloc.deallocate(allocation_ptr.cast(), rc_layout.get());
    }
}

/// # Safety
///
/// - `value_ptr` points to a valid value location within a reference-counted allocation that
///   can be described with `rc_layout` and can be deallocated with `alloc`.
/// - No access to the allocation can happen if the destructor of the returned guard gets
///   called.
#[inline]
unsafe fn deallocate_on_drop<'a, H, A>(
    alloc: &'a A,
    value_ptr: ErasedRcValuePointer<H>,
    rc_layout: RcLayout<H>,
) -> DropGuard<(), impl FnOnce(())>
where
    A: Allocator,
{
    // SAFETY: Caller guarantees the validity of all arguments.
    DropGuard::new((), move |()| unsafe { deallocate::<H, A>(alloc, value_ptr, rc_layout) })
}

/// Tries to allocate a reference-counted memory chunk for storing a value according to `rc_layout`,
/// then initializes the value with `f`. If `f` panics, the allocated memory will be deallocated.
#[inline]
pub(in crate::rcs) fn try_allocate_with_in<H, A, F>(
    alloc: &A,
    rc_layout: RcLayout<H>,
    f: F,
) -> Result<ErasedRcValuePointer<H>, AllocError>
where
    A: Allocator,
    F: FnOnce(ErasedRcValuePointer<H>),
{
    try_allocate_uninit_in::<H, A>(alloc, rc_layout).map(|ptr| {
        // SAFETY: `ptr` is allocated with `alloc` and `rc_layout`.
        let guard = unsafe { deallocate_on_drop(alloc, ptr, rc_layout) };

        f(ptr);

        // `f` did not panic, dismiss the drop guard.
        DropGuard::dismiss(guard);

        ptr
    })
}

/// Allocates a reference-counted memory chunk for storing a value according to `rc_layout`, then
/// initializes the value with `f`. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_with_in<H, A, F>(
    alloc: &A,
    rc_layout: RcLayout<H>,
    f: F,
) -> ErasedRcValuePointer<H>
where
    A: Allocator,
    F: FnOnce(ErasedRcValuePointer<H>),
{
    let value_ptr = allocate_uninit_in::<H, A>(alloc, rc_layout);

    // SAFETY: `value_ptr` is allocated with `alloc` and `rc_layout`.
    let guard = unsafe { deallocate_on_drop(alloc, value_ptr, rc_layout) };

    f(value_ptr);

    // `f` did not panic, dismiss the drop guard.
    DropGuard::dismiss(guard);

    value_ptr
}

/// Allocates a reference-counted memory chunk for storing a value according to `rc_layout`, then
/// initializes the value by copying from `source`.
///
/// # Safety
///
/// `source` must point to a location that has enough data to copy from.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) unsafe fn allocate_from_bytes_in<H, A>(
    alloc: &A,
    rc_layout: RcLayout<H>,
    source: NonNull<()>,
) -> ErasedRcValuePointer<H>
where
    A: Allocator,
{
    let ptr = allocate_uninit_in(alloc, rc_layout);

    // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
    // access to it, and caller guarantees the validify of `ptr`.
    unsafe {
        ptr::copy_nonoverlapping::<u8>(
            source.cast().as_ptr(),
            ptr.ptr.cast().as_ptr(),
            rc_layout.value_size(),
        );
    }

    ptr
}

/// Allocates a reference-counted memory chunk for storing a value according to `rc_layout`, then
/// initializes the value by copying from `value`. Note that the value bytes are duplicated but we
/// don't require `T` to be `Copy`, caller need to prevent leaks and double-frees manually.
#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_from_value_bytes_in<H, T, A>(
    alloc: &A,
    value: &T,
) -> RcValuePointer<H, T>
where
    A: Allocator,
    T: ?Sized,
{
    // SAFETY: `value` provides both layout and data source to copy from.
    let ptr = unsafe {
        allocate_from_bytes_in(alloc, RcLayout::from_value(value), NonNull::from_ref(value).cast())
    };

    ptr.unerase_with(ptr::metadata(value))
}

#[cfg(not(no_global_oom_handling))]
pub(in crate::rcs) fn allocate_from_trusted_len_in<H, A, I>(
    alloc: &A,
    iter: I,
) -> RcValuePointer<H, [I::Item]>
where
    A: Allocator,
    I: TrustedLen,
{
    /// Returns a drop guard that calls the destructors of a slice of elements on drop.
    ///
    /// # Safety
    ///
    /// - `head..tail` must describe a valid consecutive slice of `T` values when the destructor
    ///   of the returned guard is called.
    /// - After calling the returned function, the corresponding values should not be accessed
    ///   anymore.
    unsafe fn drop_range_on_drop<T>(
        head: NonNull<T>,
        tail: NonNull<T>,
    ) -> DropGuard<(NonNull<T>, NonNull<T>), impl FnOnce((NonNull<T>, NonNull<T>))> {
        // SAFETY:
        DropGuard::new((head, tail), |(head, tail)| unsafe {
            let length = tail.offset_from_unsigned(head);

            NonNull::<[T]>::slice_from_raw_parts(head, length).drop_in_place();
        })
    }

    let (length, Some(high)) = iter.size_hint() else {
        // TrustedLen contract guarantees that `upper_bound == None` implies an iterator
        // length exceeding `usize::MAX`.
        // The default implementation would collect into a vec which would panic.
        // Thus we panic here immediately without invoking `Vec` code.
        panic!("capacity overflow");
    };

    debug_assert_eq!(
        length,
        high,
        "TrustedLen iterator's size hint is not exact: {:?}",
        (length, high)
    );

    let rc_layout = RcLayout::new_array::<I::Item>(length);

    let ptr = allocate_with_in(alloc, rc_layout, |ptr| {
        let ptr = ptr.unerase::<I::Item>();

        // SAFETY: The algorithm will ensure the range's validity.
        let mut guard = unsafe { drop_range_on_drop(ptr.ptr, ptr.ptr) };

        // SAFETY: `iter` is `TrustedLen`, we can assume we will write correct number of
        // elements to the buffer.
        iter.for_each(|value| unsafe {
            guard.1.write(value);
            guard.1 = guard.1.add(1);
        });

        DropGuard::dismiss(guard);
    });

    // SAFETY: We have written `length` of `T` values to the buffer, the buffer is now
    // initialized.
    ptr.unerase_with(length)
}

/// Allocates a reference-counted memory chunk to storing values produced by `iter` using the
/// `Global` allocator.
#[cfg(not(no_global_oom_handling))]
pub(in crate::rcs) fn allocate_from_iter<H, I>(iter: I) -> RcValuePointer<H, [I::Item]>
where
    I: Iterator,
{
    trait SpecFromIter<H, I> {
        fn spec_from_iter(iter: I) -> Self;
    }

    impl<H, I> SpecFromIter<H, I> for RcValuePointer<H, [I::Item]>
    where
        I: Iterator,
    {
        default fn spec_from_iter(iter: I) -> Self {
            allocate_from_vec(iter.collect::<Vec<_>>()).0
        }
    }

    impl<H, I> SpecFromIter<H, I> for RcValuePointer<H, [I::Item]>
    where
        I: TrustedLen,
    {
        fn spec_from_iter(iter: I) -> Self {
            allocate_from_trusted_len_in(&Global, iter)
        }
    }

    RcValuePointer::spec_from_iter(iter)
}

pub(in crate::rcs) fn try_allocate_from_cloning_in<H, A, T>(
    alloc: &A,
    value: &T,
) -> Result<RcValuePointer<H, T>, AllocError>
where
    A: Allocator,
    T: CloneToUninit + ?Sized,
{
    let rc_layout =
        RcLayout::try_from_value_layout(Layout::for_value(value)).map_err(|_| AllocError)?;

    // SAFETY: `ptr` is allocated with layout calculated from `value`, we are safe to clone into it.
    try_allocate_with_in(alloc, rc_layout, |ptr| unsafe {
        value.clone_to_uninit(ptr.ptr.cast().as_ptr());
    })
    .map(|ptr| ptr.unerase_with(ptr::metadata(value)))
}

#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_from_cloning_in<H, A, T>(
    alloc: &A,
    value: &T,
) -> RcValuePointer<H, T>
where
    A: Allocator,
    T: CloneToUninit + ?Sized,
{
    // SAFETY: `ptr` is allocated with layout calculated from `value`, we are safe to clone into it.
    let ptr = allocate_with_in(alloc, RcLayout::from_value(value), |ptr| unsafe {
        value.clone_to_uninit(ptr.ptr.cast().as_ptr())
    });

    ptr.unerase_with(ptr::metadata(value))
}

#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_from_box<H, T, A>(b: Box<T, A>) -> (RcValuePointer<H, T>, A)
where
    T: ?Sized,
    A: Allocator,
{
    let ptr = allocate_from_value_bytes_in(Box::allocator(&b), &*b);
    let (box_ptr, alloc) = Box::into_raw_with_allocator(b);

    // SAFETY: Ownership of `T` is transferred into the newly allocated `ptr`, we free the old
    // `Box<T>` memory.
    unsafe { drop(Box::<ManuallyDrop<T>, &A>::from_raw_in(box_ptr as _, &alloc)) };

    (ptr, alloc)
}

#[cfg(not(no_global_oom_handling))]
#[inline]
pub(in crate::rcs) fn allocate_from_vec<H, T, A>(vec: Vec<T, A>) -> (RcValuePointer<H, [T]>, A)
where
    A: Allocator,
{
    let ptr = allocate_from_value_bytes_in(vec.allocator(), vec.as_slice());
    let (vec_ptr, _, capacity, alloc) = vec.into_parts_with_allocator();

    // SAFETY: Ownership of `[T]` is transferred into the newly allocated `ptr`, we free the old
    // `Vec<T>` memory.
    unsafe { drop(Vec::from_parts_in(vec_ptr, 0, capacity, &alloc)) };

    (ptr, alloc)
}
