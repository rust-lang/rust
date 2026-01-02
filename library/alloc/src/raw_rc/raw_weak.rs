use core::alloc::Allocator;
use core::cell::UnsafeCell;
use core::mem::{self, DropGuard};
use core::num::NonZeroUsize;
use core::ptr::{self, NonNull};

use crate::raw_rc::rc_layout::RcLayout;
use crate::raw_rc::rc_value_pointer::RcValuePointer;
use crate::raw_rc::{RefCounter, RefCounts, rc_alloc};

/// Base implementation of a weak pointer. `RawWeak` does not implement `Drop`; the user should call
/// `RawWeak::drop` or `RawWeak::drop_unchecked` manually to destroy this object.
///
/// A `RawWeak` can be either dangling or non-dangling. A dangling `RawWeak` does not point to a
/// valid value. A non-dangling `RawWeak` points to a valid reference-counted allocation. The value
/// pointed to by a `RawWeak` may be uninitialized.
pub(crate) struct RawWeak<T, A>
where
    T: ?Sized,
{
    /// Points to a (possibly uninitialized or dropped) `T` value inside of a reference-counted
    /// allocation.
    ptr: NonNull<T>,

    /// The allocator for `ptr`.
    alloc: A,
}

impl<T, A> RawWeak<T, A>
where
    T: ?Sized,
{
    pub(crate) const unsafe fn from_raw_parts(ptr: NonNull<T>, alloc: A) -> Self {
        Self { ptr, alloc }
    }

    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        A: Default,
    {
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    pub(crate) const fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) const fn as_ptr(&self) -> NonNull<T> {
        self.ptr
    }

    #[inline(never)]
    unsafe fn assume_init_drop_slow<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        let guard = unsafe { new_weak_guard::<T, A, R>(self) };

        unsafe { guard.ptr.drop_in_place() };
    }

    /// Drops the value along with the `RawWeak` object, assuming the value pointed to by `ptr` is
    /// initialized,
    #[inline]
    pub(super) unsafe fn assume_init_drop<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        if const { mem::needs_drop::<T>() } {
            unsafe { self.assume_init_drop_slow::<R>() };
        } else {
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    pub(crate) unsafe fn cast<U>(self) -> RawWeak<U, A> {
        unsafe { self.cast_with(NonNull::cast) }
    }

    #[inline]
    pub(crate) unsafe fn cast_with<U, F>(self, f: F) -> RawWeak<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        unsafe { RawWeak::from_raw_parts(f(self.ptr), self.alloc) }
    }

    /// Increments the weak count, and returns the corresponding `RawWeak` object.
    ///
    /// # Safety
    ///
    /// - `self` must only be handled by the same `RefCounter` implementation.
    #[inline]
    pub(crate) unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RefCounter,
    {
        // For reducing monomorphization cost.
        unsafe fn inner<R>(ptr: NonNull<()>)
        where
            R: RefCounter,
        {
            if let Some(value_ptr) = unsafe { try_get_rc_value_ptr(ptr) } {
                unsafe { increment_weak_ref_count::<R>(value_ptr) }
            }
        }

        unsafe {
            inner::<R>(self.ptr.cast());

            Self::from_raw_parts(self.ptr, self.alloc.clone())
        }
    }

    /// Increments the weak count, and returns the corresponding `RawWeak` object, assuming `self`
    /// is non-dangling.
    ///
    /// # Safety
    ///
    /// - `self` must only be handled by the same `RefCounter` implementation.
    /// - `self` is non-dangling.
    pub(crate) unsafe fn clone_unchecked<R>(&self) -> Self
    where
        A: Clone,
        R: RefCounter,
    {
        unsafe {
            increment_weak_ref_count::<R>(self.value_ptr_unchecked());

            Self::from_raw_parts(self.ptr, self.alloc.clone())
        }
    }

    /// Drops this weak pointer.
    #[inline]
    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        if !is_dangling(self.ptr.cast()) {
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    /// Drops this weak pointer, assuming `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    #[inline]
    pub(super) unsafe fn drop_unchecked<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees `self` is non-dangling, so `self.ptr` must point to the value
        // location in a valid reference-counted allocation.
        let value_ptr = unsafe { self.value_ptr_unchecked() };

        let is_last_weak_ref = unsafe { decrement_weak_ref_count::<R>(value_ptr) };

        if is_last_weak_ref {
            let rc_layout = unsafe { RcLayout::from_value_ptr_unchecked(self.ptr) };

            unsafe { rc_alloc::deallocate::<A>(value_ptr, &self.alloc, rc_layout) }
        }
    }

    pub(crate) fn into_raw(self) -> NonNull<T> {
        self.ptr
    }

    pub(crate) fn into_raw_parts(self) -> (NonNull<T>, A) {
        (self.ptr, self.alloc)
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    pub(crate) fn ptr_ne(&self, other: &Self) -> bool {
        !ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the `RefCounts` object inside the reference-counted allocation, assume `self` is
    /// non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    #[cfg(not(no_global_oom_handling))]
    pub(super) unsafe fn ref_counts_unchecked(&self) -> &RefCounts {
        unsafe { self.value_ptr_unchecked().ref_counts_ptr().as_ref() }
    }

    /// Returns the strong reference count object inside the reference-counted allocation if `self`
    /// is non-dangling.
    pub(crate) fn strong_count(&self) -> Option<&UnsafeCell<usize>> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.strong_count_unchecked() })
    }

    /// Returns the strong reference count object inside the reference-counted allocation, assume
    /// `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    pub(super) unsafe fn strong_count_unchecked(&self) -> &UnsafeCell<usize> {
        unsafe { self.value_ptr_unchecked().strong_count_ptr().as_ref() }
    }

    /// Returns the weak reference count object inside the reference-counted allocation if `self` is
    /// non-dangling.
    pub(crate) fn weak_count(&self) -> Option<&UnsafeCell<usize>> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.weak_count_unchecked() })
    }

    /// Returns the weak reference count object inside the reference-counted allocation, assume
    /// `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    pub(super) unsafe fn weak_count_unchecked(&self) -> &UnsafeCell<usize> {
        unsafe { self.value_ptr_unchecked().weak_count_ptr().as_ref() }
    }

    /// Sets the contained pointer to a new value.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer to a value object that lives in a reference-counted
    ///   allocation.
    /// - The allocation can be deallocated with the associated allocator.
    #[cfg(not(no_global_oom_handling))]
    pub(super) unsafe fn set_ptr(&mut self, ptr: NonNull<T>) {
        self.ptr = ptr;
    }

    /// Returns a pointer to the value location of the reference-counted allocation, assume `self`
    /// is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    #[inline]
    pub(super) unsafe fn value_ptr_unchecked(&self) -> RcValuePointer {
        // SAFETY: Caller guarantees `self` is non-dangling, so `self.ptr` must point to the value
        // location in a valid reference-counted allocation.
        unsafe { RcValuePointer::from_value_ptr(self.ptr.cast()) }
    }
}

// We choose `NonZeroUsize::MAX` as the address for dangling weak pointers because:
//
// - It does not point to any object that is stored inside a reference-counted allocation. Because
//   otherwise the corresponding `RefCounts` object will be placed at
//  `NonZeroUsize::MAX - size_of::<RefCounts>()`, which is an odd number that violates `RefCounts`'s
//   alignment requirement.
// - All bytes in the byte representation of `NonZeroUsize::MAX` are the same, which makes it
//   possible to utilize `memset` in certain situations like creating an array of dangling weak
//   pointers.
const DANGLING_WEAK_ADDRESS: NonZeroUsize = {
    let address = NonZeroUsize::MAX;

    // Verifies that `address` must not be a valid address in a reference-counted allocation so it
    // can be safely used as the dangling pointer address.
    assert!(address.get().wrapping_sub(size_of::<RefCounts>()) % align_of::<RefCounts>() != 0);

    address
};

#[inline]
fn is_dangling(value_ptr: NonNull<()>) -> bool {
    value_ptr.addr() == DANGLING_WEAK_ADDRESS
}

/// # Safety
///
/// Either `is_dangling(dangling_or_value_ptr)`, or `dangling_or_value_ptr` has a valid address for
/// the value location of a reference-counted allocation.
#[inline]
unsafe fn try_get_rc_value_ptr(dangling_or_value_ptr: NonNull<()>) -> Option<RcValuePointer> {
    if is_dangling(dangling_or_value_ptr) {
        None
    } else {
        // SAFETY: We have checked `dangling_or_value_ptr` not being dangling, and caller guarantees
        // the validity of `dangling_or_value_ptr`.

        Some(unsafe { RcValuePointer::from_value_ptr(dangling_or_value_ptr) })
    }
}

/// Decrements weak reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to the value location within a valid reference-counted allocation.
/// - The corresponding weak count must not be zero.
#[inline]
unsafe fn decrement_weak_ref_count<R>(value_ptr: RcValuePointer) -> bool
where
    R: RefCounter,
{
    unsafe { R::from_raw_counter(value_ptr.weak_count_ptr().as_ref()) }.decrement()
}

/// Increments weak reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
///
/// # Safety
///
/// `value_ptr` must point to the value location within a valid reference-counted allocation.
#[inline]
unsafe fn increment_weak_ref_count<R>(value_ptr: RcValuePointer)
where
    R: RefCounter,
{
    unsafe { R::from_raw_counter(value_ptr.weak_count_ptr().as_ref()) }.increment()
}

/// Creates a drop guard that calls `RawWeak::drop_unchecked` on drop.
///
/// # Safety
///
/// - `weak` is non-dangling.
/// - After the returned `DropGuard` being dropped, the allocation pointed to by the weak pointer
///   must not be accessed anymore.
/// - All accesses to `weak` must use the same `R` for `RefCounter`.
pub(super) unsafe fn new_weak_guard<'a, T, A, R>(
    weak: &'a mut RawWeak<T, A>,
) -> DropGuard<&'a mut RawWeak<T, A>, impl FnOnce(&'a mut RawWeak<T, A>)>
where
    T: ?Sized,
    A: Allocator,
    R: RefCounter,
{
    // SAFETY: Caller guarantees that `weak` is non-dangling and the corresponding allocation will
    // not be accessed after dropping.
    DropGuard::new(weak, |weak| unsafe { weak.drop_unchecked::<R>() })
}
