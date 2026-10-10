//! Provides infrastructure for reference counted types.
//!
//! A reference counted allocation contains the following parts in order:
//!
//! - Padding: allows header and value to be arranged consecutively while being properly aligned.
//!   We choose the padding size to be
//!   `size_of::<Header>().next_multiple_of(align_of::<Value>()) - size_of::<Header>()`.
//! - Header: stores reference counters.
//! - Value: the actual contained value.
//!
//! We use the following layout for the allocation:
//!
//! - Alignment: `align_of::<Header>().max(align_of::<Value>())`.
//! - Size: `size_of::<Header>().next_multiple_of(align_of::<Value>()) + size_of::<Value>()`.
//!
//! Note that it is possible that the size of an allocation is not a multiple of its alignment.

use core::alloc::{AllocError, Allocator};
use core::clone::CloneToUninit;
#[cfg(not(no_global_oom_handling))]
use core::iter::TrustedLen;
use core::mem::DropGuard;
#[cfg(not(no_global_oom_handling))]
use core::mem::ManuallyDrop;
use core::ptr;
#[cfg(not(no_global_oom_handling))]
use core::ptr::NonNull;

pub(super) use rc_value_pointer::{ErasedRcValuePointer, RcValuePointer};

#[cfg(not(no_global_oom_handling))]
use crate::alloc::{self, Global};
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
use crate::rcs::common::rc_layout::RcLayout;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

mod rc_layout;
mod rc_value_pointer;

// - `repr(C)` is required for deterministic reference counter layout, which is required in debugger
//   visualizers.
// - `align(2)` is required to ensure `usize::MAX` can be used as the dangling pointer address.
#[repr(C, align(2))]
pub(super) struct Header<C> {
    pub(super) strong: C,
    pub(super) weak: C,
}

impl<C> Header<C> {
    #[inline]
    pub(super) fn try_allocate_uninit_for_type<A, T>(
        alloc: &A,
    ) -> Result<RcValuePointer<T>, AllocError>
    where
        A: Allocator,
    {
        Ok(try_allocate_uninit(alloc, RcLayout::new::<Self, T>())?.unerase())
    }

    #[inline]
    pub(super) fn try_allocate_zeroed_for_type<A, T>(
        alloc: &A,
    ) -> Result<RcValuePointer<T>, AllocError>
    where
        A: Allocator,
    {
        Ok(try_allocate_zeroed(alloc, RcLayout::new::<Self, T>())?.unerase())
    }

    pub(super) fn try_allocate_from_cloning<A, T>(
        alloc: &A,
        value: &T,
    ) -> Result<RcValuePointer<T>, AllocError>
    where
        A: Allocator,
        T: CloneToUninit + ?Sized,
    {
        let Ok(rc_layout) = RcLayout::try_from_value::<Self, T>(value) else {
            return Err(AllocError);
        };

        // SAFETY: `ptr` is allocated with layout calculated from `value`, we are safe to clone into it.
        let ptr = try_allocate_with(alloc, rc_layout, |ptr| unsafe {
            value.clone_to_uninit(ptr.ptr.cast().as_ptr());
        })?;

        Ok(ptr.unerase_with(ptr::metadata(value)))
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn allocate_uninit_for_type<A, T>(alloc: &A) -> RcValuePointer<T>
    where
        A: Allocator,
    {
        allocate_uninit(alloc, RcLayout::new::<Self, T>()).unerase()
    }

    /// # Safety
    ///
    /// Caller must ensure that `T` is small enough to store inside a reference counted allocation
    /// with header of type `Self`.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) unsafe fn allocate_uninit_for_value_unchecked<A, T>(
        alloc: &A,
        value: &T,
    ) -> RcValuePointer<T>
    where
        A: Allocator,
        T: ?Sized,
    {
        // SAFETY: Caller guarantees we are safe to calculate the allocation layout.
        allocate_uninit(alloc, unsafe { RcLayout::from_value_unchecked::<Self, T>(value) })
            .unerase_with(ptr::metadata(value))
    }

    #[cfg(not(no_global_oom_handling))]
    pub(super) fn allocate_uninit_slice<A, T>(alloc: &A, length: usize) -> RcValuePointer<[T]>
    where
        A: Allocator,
    {
        allocate_uninit(alloc, RcLayout::new_array::<Self, T>(length)).unerase_with(length)
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn allocate_zeroed_for_type<A, T>(alloc: &A) -> RcValuePointer<T>
    where
        A: Allocator,
    {
        allocate_zeroed(alloc, RcLayout::new::<Self, T>()).unerase()
    }

    #[cfg(not(no_global_oom_handling))]
    pub(super) fn allocate_zeroed_slice<A, T>(alloc: &A, length: usize) -> RcValuePointer<[T]>
    where
        A: Allocator,
    {
        allocate_zeroed(alloc, RcLayout::new_array::<Self, T>(length)).unerase_with(length)
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn allocate_from_cloning<A, T>(alloc: &A, value: &T) -> RcValuePointer<T>
    where
        A: Allocator,
        T: CloneToUninit + ?Sized,
    {
        // SAFETY: `ptr` is allocated with layout calculated from `value`, we are safe to clone into it.
        let ptr = allocate_with(alloc, RcLayout::from_value::<Self, T>(value), |ptr| unsafe {
            value.clone_to_uninit(ptr.ptr.cast().as_ptr())
        });

        ptr.unerase_with(ptr::metadata(value))
    }

    #[cfg(not(no_global_oom_handling))]
    pub(super) fn allocate_from_default<A, T>(alloc: &A) -> RcValuePointer<T>
    where
        A: Allocator,
        T: Default,
    {
        // SAFETY: We have exclusive access to `ptr`.
        allocate_with(alloc, RcLayout::new::<Self, T>(), |ptr| unsafe {
            ptr.unerase().ptr.write(T::default())
        })
        .unerase()
    }

    /// Allocates a reference-counted memory chunk for storing a value according to `rc_layout`,
    /// then initializes the value by copying from `value`. Note that the value bytes are duplicated
    /// but we don't require `T` to be `Copy`, the caller needs to prevent leaks and double-frees
    /// manually.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn allocate_from_value_bytes<T, A>(alloc: &A, value: &T) -> RcValuePointer<T>
    where
        A: Allocator,
        T: ?Sized,
    {
        // SAFETY: `value` provides both layout and data source to copy from.
        let ptr = unsafe {
            allocate_from_bytes(
                alloc,
                RcLayout::from_value::<Self, T>(value),
                NonNull::from_ref(value).cast(),
            )
        };

        ptr.unerase_with(ptr::metadata(value))
    }

    #[cfg(not(no_global_oom_handling))]
    pub(super) fn allocate_from_trusted_len<A, I>(alloc: &A, iter: I) -> RcValuePointer<[I::Item]>
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
        ///   any more.
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

        let rc_layout = RcLayout::new_array::<Self, I::Item>(length);

        let ptr = allocate_with(alloc, rc_layout, |ptr| {
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

        ptr.unerase_with(length)
    }

    /// Allocates a reference-counted memory chunk to store values produced by `iter` using the
    /// `Global` allocator.
    #[cfg(not(no_global_oom_handling))]
    pub(super) fn allocate_from_iter<I>(iter: I) -> RcValuePointer<[I::Item]>
    where
        I: Iterator,
    {
        trait SpecAllocate: Iterator {
            fn spec_allocate<C>(self) -> RcValuePointer<[Self::Item]>;
        }

        impl<I> SpecAllocate for I
        where
            I: Iterator,
        {
            default fn spec_allocate<C>(self) -> RcValuePointer<[Self::Item]> {
                Header::<C>::allocate_from_vec(self.collect::<Vec<_>>()).0
            }
        }

        impl<I> SpecAllocate for I
        where
            I: TrustedLen,
        {
            fn spec_allocate<C>(self) -> RcValuePointer<[Self::Item]> {
                Header::<C>::allocate_from_trusted_len(&Global, self)
            }
        }

        iter.spec_allocate::<C>()
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn allocate_from_box<T, A>(b: Box<T, A>) -> (RcValuePointer<T>, A)
    where
        T: ?Sized,
        A: Allocator,
    {
        let ptr = Self::allocate_from_value_bytes(Box::allocator(&b), &*b);
        let (box_ptr, alloc) = Box::into_raw_with_allocator(b);

        // SAFETY: Ownership of `T` is transferred into the newly allocated `ptr`, we free the old
        // `Box<T>` memory.
        unsafe { drop(Box::<ManuallyDrop<T>, &A>::from_raw_in(box_ptr as _, &alloc)) };

        (ptr, alloc)
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(super) fn allocate_from_vec<T, A>(vec: Vec<T, A>) -> (RcValuePointer<[T]>, A)
    where
        A: Allocator,
    {
        let ptr = Self::allocate_from_value_bytes(vec.allocator(), vec.as_slice());
        let (vec_ptr, _, capacity, alloc) = vec.into_parts_with_allocator();

        // SAFETY: Ownership of `[T]` is transferred into the newly allocated `ptr`, we free the old
        // `Vec<T>` memory.
        unsafe { drop(Vec::from_parts_in(vec_ptr, 0, capacity, &alloc)) };

        (ptr, alloc)
    }

    /// # Safety
    ///
    /// - `ptr` must point to a valid reference counted allocation with header of type `Self`.
    /// - Caller must ensure there is no further access to the allocation after calling `deallocate`.
    #[inline]
    pub(super) unsafe fn deallocate<A, T>(alloc: &A, ptr: RcValuePointer<T>)
    where
        A: Allocator,
        T: ?Sized,
    {
        // SAFETY: Caller guarantees the validity of `ptr`.
        unsafe {
            deallocate(alloc, ptr.erase(), RcLayout::from_value_ptr_unchecked::<Self, T>(ptr.ptr));
        }
    }

    /// # Safety
    ///
    /// `ptr` must point to a valid reference counted allocation containing an initialized header of
    /// type `Self`.
    #[inline]
    pub(super) unsafe fn get<'a>(ptr: ErasedRcValuePointer) -> &'a Self {
        // SAFETY: Caller guarantees the validity of `ptr`.
        unsafe { ptr.ptr.cast().sub(1).as_ref() }
    }

    /// # Safety
    ///
    /// `ptr` must point to a valid reference counted allocation that can store a header of type
    /// `Self`.
    #[inline]
    pub(super) unsafe fn write_to(self, ptr: ErasedRcValuePointer) {
        // SAFETY: Caller guarantees the validity of `ptr`.
        unsafe { ptr.ptr.cast().sub(1).write(self) };
    }
}

/// Tries to allocate uninitialized memory for a reference-counted allocation using allocator
/// `alloc` and layout `rc_layout`. Returns a pointer to the value location.
#[inline]
fn try_allocate_uninit<A>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<ErasedRcValuePointer, AllocError>
where
    A: Allocator,
{
    let allocation_ptr = alloc.allocate(rc_layout.layout())?;

    // SAFETY: `allocation_ptr` is allocated with `rc_layout`, so we can safely acquire a pointer to
    // the value.
    Ok(ErasedRcValuePointer::new(unsafe {
        allocation_ptr.cast().byte_add(rc_layout.value_offset())
    }))
}

/// Tries to allocate zeroed memory for a reference-counted allocation using allocator
/// `alloc` and layout `rc_layout`. Returns a pointer to the value location.
#[inline]
fn try_allocate_zeroed<A>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<ErasedRcValuePointer, AllocError>
where
    A: Allocator,
{
    let allocation_ptr = alloc.allocate_zeroed(rc_layout.layout())?;

    // SAFETY: `allocation_ptr` is allocated with `rc_layout`, so we can safely acquire a pointer to
    // the value.
    Ok(ErasedRcValuePointer::new(unsafe {
        allocation_ptr.cast().byte_add(rc_layout.value_offset())
    }))
}

/// Tries to allocate a reference-counted memory chunk for storing a value according to `rc_layout`,
/// then initializes the value with `f`. If `f` panics, the allocated memory will be deallocated.
#[inline]
fn try_allocate_with<A, F>(
    alloc: &A,
    rc_layout: RcLayout,
    f: F,
) -> Result<ErasedRcValuePointer, AllocError>
where
    A: Allocator,
    F: FnOnce(ErasedRcValuePointer),
{
    let ptr = try_allocate_uninit(alloc, rc_layout)?;

    // SAFETY: `ptr` is allocated with `alloc` and `rc_layout`.
    let guard = unsafe { deallocate_on_drop(alloc, ptr, rc_layout) };

    f(ptr);

    // `f` did not panic, dismiss the drop guard.
    DropGuard::dismiss(guard);

    Ok(ptr)
}

/// Allocates uninitialized memory for a reference-counted allocation using allocator `alloc` and
/// layout `rc_layout`. Returns a pointer to the value location. If the allocation fails, panic will
/// be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_uninit<A>(alloc: &A, rc_layout: RcLayout) -> ErasedRcValuePointer
where
    A: Allocator,
{
    match try_allocate_uninit(alloc, rc_layout) {
        Ok(result) => result,
        Err(_) => alloc::handle_alloc_error(rc_layout.layout()),
    }
}

/// Allocates zeroed memory for a reference-counted allocation using allocator `alloc` and
/// layout `rc_layout`. Returns a pointer to the value location. If the allocation fails, panic will
/// be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_zeroed<A>(alloc: &A, rc_layout: RcLayout) -> ErasedRcValuePointer
where
    A: Allocator,
{
    match try_allocate_zeroed(alloc, rc_layout) {
        Ok(result) => result,
        Err(_) => alloc::handle_alloc_error(rc_layout.layout()),
    }
}

/// Allocates a reference-counted memory chunk for storing a value according to `rc_layout`, then
/// initializes the value with `f`. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_with<A, F>(alloc: &A, rc_layout: RcLayout, f: F) -> ErasedRcValuePointer
where
    A: Allocator,
    F: FnOnce(ErasedRcValuePointer),
{
    let value_ptr = allocate_uninit(alloc, rc_layout);

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
unsafe fn allocate_from_bytes<A>(
    alloc: &A,
    rc_layout: RcLayout,
    source: NonNull<()>,
) -> ErasedRcValuePointer
where
    A: Allocator,
{
    let ptr = allocate_uninit(alloc, rc_layout);

    // SAFETY: `ptr` points to a newly allocated reference counted allocation, we have exclusive
    // access to it, and caller guarantees the validity of `ptr`.
    unsafe {
        ptr::copy_nonoverlapping::<u8>(
            source.cast().as_ptr(),
            ptr.ptr.cast().as_ptr(),
            rc_layout.value_size(),
        );
    }

    ptr
}

/// Deallocates a reference-counted allocation with a value object pointed to by `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` points to a valid reference-counted allocation that was allocated using
///   `rc_layout`.
#[inline]
unsafe fn deallocate<A>(alloc: &A, value_ptr: ErasedRcValuePointer, rc_layout: RcLayout)
where
    A: Allocator,
{
    let value_offset = rc_layout.value_offset();

    // SAFETY: Caller guarantees the validity of `value_ptr`.
    unsafe {
        let allocation_ptr = value_ptr.ptr.byte_sub(value_offset);

        alloc.deallocate(allocation_ptr.cast(), rc_layout.layout());
    }
}

/// # Safety
///
/// - `value_ptr` points to a valid value location within a reference-counted allocation that
///   can be described with `rc_layout` and can be deallocated with `alloc`.
/// - No access to the allocation can happen if the destructor of the returned guard gets
///   called.
#[inline]
unsafe fn deallocate_on_drop<'a, A>(
    alloc: &'a A,
    value_ptr: ErasedRcValuePointer,
    rc_layout: RcLayout,
) -> DropGuard<(), impl FnOnce(())>
where
    A: Allocator,
{
    // SAFETY: Caller guarantees the validity of all arguments.
    DropGuard::new((), move |()| unsafe { deallocate::<A>(alloc, value_ptr, rc_layout) })
}
