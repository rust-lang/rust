use core::alloc::{AllocError, Allocator, AllocatorClone};
use core::cell::UnsafeCell;
use core::fmt::{self, Debug, Formatter};
use core::marker::Unsize;
use core::mem::{self, DropGuard};
use core::num::NonZeroUsize;
use core::ops::{CoerceUnsized, DispatchFromDyn};
use core::ptr::{self, NonNull};

use crate::alloc::Global;
use crate::raw_rc::raw_rc::RawRc;
use crate::raw_rc::rc_layout::{RcLayout, RcLayoutExt};
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
        // SAFETY: Caller guarantees the validity of `ptr`.
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    pub(crate) const fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) const fn as_ptr(&self) -> NonNull<T> {
        self.ptr
    }

    /// # Safety
    ///
    /// - `self` is non-dangling.
    /// - `self` is only accessed with the same `R`.
    /// - The contained value is properly initialized.
    /// - `self` has exclusive ownership of the contained value.
    /// - No further access to the allocation through `self` after calling `assume_init_drop_slow`.
    #[inline(never)]
    unsafe fn assume_init_drop_slow<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees `self` is non-dangling and will not further access the
        // allocation through `self`.
        let guard = unsafe { new_weak_guard::<T, A, R>(self) };

        // SAFETY: Caller guarantees that the value is initialized and will not further access the
        // allocation through `self`.
        unsafe { guard.ptr.drop_in_place() };
    }

    /// Drops the value along with the `RawWeak` object, assuming the value pointed to by `ptr` is
    /// initialized,
    ///
    /// # Safety
    ///
    /// - `self` is non-dangling.
    /// - `self` is only accessed with the same `R`.
    /// - The contained value is properly initialized.
    /// - `self` has exclusive ownership of the contained value.
    /// - No further access to the allocation through `self` after calling `assume_init_drop_slow`.
    #[inline]
    pub(super) unsafe fn assume_init_drop<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees the safety contract holds.
        unsafe {
            if const { mem::needs_drop::<T>() } {
                self.assume_init_drop_slow::<R>();
            } else {
                self.drop_unchecked::<R>();
            }
        }
    }

    pub(crate) unsafe fn cast<U>(self) -> RawWeak<U, A> {
        // SAFETY: Caller guarantees `T` can be reinterpreted as `U`.
        unsafe { self.cast_with(NonNull::cast) }
    }

    #[inline]
    pub(crate) unsafe fn cast_with<U, F>(self, f: F) -> RawWeak<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        // SAFETY: Caller guarantees `T` can be reinterpreted as `U` and `f` returns a reinterpreted
        // pointer to the contained value.
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
        A: AllocatorClone,
        R: RefCounter,
    {
        // For reducing monomorphization cost.
        unsafe fn inner<R>(ptr: NonNull<()>)
        where
            R: RefCounter,
        {
            // SAFETY: Caller guarantees the validify of `ptr`.
            if let Some(value_ptr) = unsafe { try_get_rc_value_ptr(ptr) } {
                // SAFETY: `value_ptr` is guaranteed to be non-dangling on success.
                unsafe { increment_weak_ref_count::<R>(value_ptr) }
            }
        }

        // SAFETY: `self.ptr` is guaranteed to be either dangling or points to a valid allocation.
        // When `inner` returns, strong count has been incremented if `self.ptr` is non-dangling, we
        // can take ownership of the new reference count.
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
        A: AllocatorClone,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees `self` is non-dangling.
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
            // SAFETY: We have checked `self` is non-dangling, and caller guarantees no further
            // access to the allocation.
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    /// Drops this weak pointer, assuming `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// - `self` is non-dangling.
    /// - No further access to the allocation through `self` after calling `drop_unchecked`.
    #[inline]
    pub(super) unsafe fn drop_unchecked<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees `self` is non-dangling.
        let value_ptr = unsafe { self.value_ptr_unchecked() };

        // SAFETY: `value_ptr` is guaranteed to be valid.
        let is_last_weak_ref = unsafe { decrement_weak_ref_count::<R>(value_ptr) };

        if is_last_weak_ref {
            // SAFETY: `self.ptr` lives inside `RawWeak`, so there must be a valid `RcLayout` for
            // it.
            let rc_layout = unsafe { RcLayout::from_value_ptr_unchecked(self.ptr) };

            // SAFETY: We own the last weak reference, so it is valid to deallocate the allocation.
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
        ptr::eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    pub(crate) fn ptr_ne(&self, other: &Self) -> bool {
        !ptr::eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the `RefCounts` object inside the reference-counted allocation, assume `self` is
    /// non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    #[cfg(not(no_global_oom_handling))]
    pub(super) unsafe fn ref_counts_unchecked(&self) -> &RefCounts {
        // SAFETY: Caller guarantees `self` is non-dangling.
        unsafe { self.value_ptr_unchecked().ref_counts_ptr().as_ref() }
    }

    /// Returns the strong reference count object inside the reference-counted allocation if `self`
    /// is non-dangling.
    pub(crate) fn strong_count(&self) -> Option<&UnsafeCell<usize>> {
        // SAFETY: We have checked `self` is non-dangling.
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.strong_count_unchecked() })
    }

    /// Returns the strong reference count object inside the reference-counted allocation, assume
    /// `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    pub(super) unsafe fn strong_count_unchecked(&self) -> &UnsafeCell<usize> {
        // SAFETY: Caller guarantees `self` is non-dangling.
        unsafe { self.value_ptr_unchecked().strong_count_ptr().as_ref() }
    }

    /// # Safety
    ///
    /// `ptr` must point to a valid value storage of a reference counted allocation.
    pub(crate) unsafe fn strong_count_from_raw_unchecked<'a>(
        ptr: NonNull<T>,
    ) -> &'a UnsafeCell<usize> {
        // SAFETY: Caller guarantees the validity of `ptr`.
        unsafe { RcValuePointer::from_value_ptr(ptr.cast()).strong_count_ptr().as_ref() }
    }

    /// Returns the weak reference count object inside the reference-counted allocation if `self` is
    /// non-dangling.
    pub(crate) fn weak_count(&self) -> Option<&UnsafeCell<usize>> {
        // SAFETY: We have checked `self` is non-dangling.
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.weak_count_unchecked() })
    }

    /// Returns the weak reference count object inside the reference-counted allocation, assume
    /// `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    pub(super) unsafe fn weak_count_unchecked(&self) -> &UnsafeCell<usize> {
        // SAFETY: Caller guarantees `self` is non-dangling.
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

    /// Creates a `RawRc` object if there are non-zero strong reference counts.
    ///
    /// # Safety
    ///
    /// `self` must only be handled by the same `RefCounter` implementation.
    pub(crate) unsafe fn upgrade<R>(&self) -> Option<RawRc<T, A>>
    where
        A: AllocatorClone,
        R: RefCounter,
    {
        // For reducing monomorphization cost.
        unsafe fn inner<R>(value_ptr: NonNull<()>) -> bool
        where
            R: RefCounter,
        {
            // SAFETY: Caller guarantees the validify of `value_ptr`.
            if let Some(value_ptr) = unsafe { try_get_rc_value_ptr(value_ptr) } {
                // SAFETY: `value_ptr` is guaranteed to be valid.
                unsafe { R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()) }.try_upgrade()
            } else {
                false
            }
        }

        // SAFETY: Caller guarantees the consistency of `R`.
        let upgraded = unsafe { inner::<R>(self.ptr.cast()) };

        if upgraded {
            // SAFETY: The strong count has been incremented, we can take ownership of it.
            Some(unsafe { RawRc::from_raw_parts(self.ptr, self.alloc.clone()) })
        } else {
            None
        }
    }

    /// Returns a pointer to the value location of the reference-counted allocation, assume `self`
    /// is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    #[inline]
    pub(super) unsafe fn value_ptr_unchecked(&self) -> RcValuePointer {
        // SAFETY: Caller guarantees `self` is non-dangling.
        unsafe { RcValuePointer::from_value_ptr(self.ptr.cast()) }
    }
}

impl<T, A> RawWeak<T, A> {
    pub(crate) const fn new_dangling_in(alloc: A) -> Self {
        // SAFETY: It is valid to create a dangling `RawWeak`.
        unsafe { Self::from_raw_parts(NonNull::without_provenance(DANGLING_WEAK_ADDRESS), alloc) }
    }

    pub(crate) fn new_dangling() -> Self
    where
        A: Default,
    {
        Self::new_dangling_in(A::default())
    }

    pub(crate) fn try_new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        rc_alloc::try_allocate_uninit_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT)
            // SAFETY: `ptr` is allocated with `alloc`.
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) })
    }

    pub(crate) fn try_new_uninit<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        rc_alloc::try_allocate_uninit::<A, STRONG_COUNT>(T::RC_LAYOUT)
            // SAFETY: `ptr` is allocated with `alloc`.
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) })
    }

    pub(crate) fn try_new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        rc_alloc::try_allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT)
            // SAFETY: `ptr` is allocated with `alloc`.
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) })
    }

    pub(crate) fn try_new_zeroed<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        rc_alloc::try_allocate_zeroed::<A, STRONG_COUNT>(T::RC_LAYOUT)
            // SAFETY: `ptr` is allocated with `alloc`.
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        let ptr = rc_alloc::allocate_uninit_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT);

        // SAFETY: `ptr` is allocated with `alloc`.
        unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        let (ptr, alloc) = rc_alloc::allocate_uninit::<A, STRONG_COUNT>(T::RC_LAYOUT);

        // SAFETY: `ptr` is allocated with `alloc`.
        unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        let ptr = rc_alloc::allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT);

        // SAFETY: `ptr` is allocated with `alloc`.
        unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        let (ptr, alloc) = rc_alloc::allocate_zeroed::<A, STRONG_COUNT>(T::RC_LAYOUT);

        // SAFETY: `ptr` is allocated with `alloc`.
        unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) }
    }

    /// Consumes the `RawWeak` object and returns the contained value, assuming the value is
    /// initialized.
    ///
    /// # Safety
    ///
    /// - `self` is non-dangling.
    /// - The value pointed to by `self` is initialized.
    /// - The strong reference count is zero.
    pub(super) unsafe fn assume_init_into_inner<R>(mut self) -> T
    where
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees the strong count is zero and the value is initialized, we have
        // exclusive ownership of the contained value, we can transfer its ownership to caller.
        unsafe {
            let result = self.ptr.read();

            self.drop_unchecked::<R>();

            result
        }
    }
}

impl<T, A> RawWeak<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    fn allocate_in<F>(length: usize, alloc: A, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(&A, RcLayout) -> RcValuePointer,
    {
        let rc_layout = RcLayout::new_array::<T>(length);
        let ptr = allocate_fn(&alloc, rc_layout);

        // SAFETY: `ptr` is allocated with `alloc`.
        unsafe {
            Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.as_ptr().cast(), length), alloc)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    fn allocate<F>(length: usize, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(RcLayout) -> (RcValuePointer, A),
    {
        let rc_layout = RcLayout::new_array::<T>(length);
        let (ptr, alloc) = allocate_fn(rc_layout);

        // SAFETY: `ptr` is allocated with `alloc`.
        unsafe {
            Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.as_ptr().cast(), length), alloc)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, rc_alloc::allocate_uninit_in::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::allocate(length, rc_alloc::allocate_uninit::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, rc_alloc::allocate_zeroed_in::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::allocate(length, rc_alloc::allocate_zeroed::<A, STRONG_COUNT>)
    }
}

impl<T, U, A> CoerceUnsized<RawWeak<U, A>> for RawWeak<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Debug for RawWeak<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("(Weak)")
    }
}

impl<T, A> Default for RawWeak<T, A>
where
    A: Default,
{
    fn default() -> Self {
        Self::new_dangling()
    }
}

impl<T, U> DispatchFromDyn<RawWeak<U, Global>> for RawWeak<T, Global>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
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
    // SAFETY: Caller guarantees the validify of `value_ptr` and the consistency of `R`.
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
    // SAFETY: Caller guarantees the validify of `value_ptr` and the consistency of `R`.
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
