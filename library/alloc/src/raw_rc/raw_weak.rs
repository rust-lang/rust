use core::alloc::{AllocError, Allocator};
use core::cell::UnsafeCell;
use core::fmt::{self, Debug, Formatter};
use core::marker::{PhantomData, Unsize};
use core::mem;
use core::num::NonZeroUsize;
use core::ops::{CoerceUnsized, Deref, DerefMut, DispatchFromDyn};
use core::ptr::{self, NonNull};

use crate::alloc::Global;
use crate::raw_rc::raw_rc::RawRc;
use crate::raw_rc::rc_layout::{RcLayout, RcLayoutExt};
use crate::raw_rc::{RcOps, RefCounts};

// We choose `NonZeroUsize::MAX` as the address for dangling weak pointers because:
//
// - It does not point to any object that is stored inside a reference counted allocation. Because
//   otherwise the corresponding `RefCounts` object will be placed at
//   `NonZeroUsize::MAX - size_of::<RefCounts>()`, which is an odd number that violates
//   `RefCounts`'s alignment requirement.
// - All bytes in the byte representation of `NonZeroUsize::MAX` are the same, which makes it
//   possible to utilize `memset` in certain situations like creating an array of dangling weak
//   pointers.
const DANGLING_WEAK_ADDRESS: NonZeroUsize = NonZeroUsize::MAX;

// Verify that `DANGLING_WEAK_ADDRESS` is a suitable address for dangling weak pointers.
const _: () = assert!(
    DANGLING_WEAK_ADDRESS.get().wrapping_sub(size_of::<RefCounts>()) % align_of::<RefCounts>() != 0
);

#[inline]
fn is_dangling(value_ptr: NonNull<()>) -> bool {
    value_ptr.addr() == DANGLING_WEAK_ADDRESS
}

/// Decrements weak reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
unsafe fn decrement_weak_ref_count<R>(value_ptr: NonNull<()>) -> bool
where
    R: RcOps,
{
    unsafe { R::decrement_ref_count(super::weak_count_ptr_from_value_ptr(value_ptr).as_ref()) }
}

/// Increments weak reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
unsafe fn increment_weak_ref_count<R>(value_ptr: NonNull<()>)
where
    R: RcOps,
{
    unsafe { R::increment_ref_count(super::weak_count_ptr_from_value_ptr(value_ptr).as_ref()) }
}

/// Calls `RawWeak::drop_unchecked` on drop.
pub(super) struct WeakGuard<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    weak: &'a mut RawWeak<T, A>,
    _phantom_data: PhantomData<R>,
}

impl<'a, T, A, R> WeakGuard<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    /// # Safety
    ///
    /// - `weak` is non-dangling.
    /// - After `WeakGuard` being dropped, the allocation pointed by the weak pointer should not be
    ///   accessed anymore.
    pub(super) unsafe fn new(weak: &'a mut RawWeak<T, A>) -> Self {
        Self { weak, _phantom_data: PhantomData }
    }
}

impl<T, A, R> Deref for WeakGuard<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    type Target = RawWeak<T, A>;

    fn deref(&self) -> &Self::Target {
        &*self.weak
    }
}

impl<T, A, R> DerefMut for WeakGuard<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.weak
    }
}

impl<T, A, R> Drop for WeakGuard<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    fn drop(&mut self) {
        unsafe { self.weak.drop_unchecked::<R>() };
    }
}

/// Base implementation of a weak pointer. `RawWeak` does not implement `Drop`, user should call
/// `RawWeak::drop` or `RawWeak::drop_unchecked` manually to drop this object.
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

    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) fn as_ptr(&self) -> NonNull<T> {
        self.ptr
    }

    #[inline(never)]
    unsafe fn assume_init_drop_slow<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        let guard = unsafe { WeakGuard::<T, A, R>::new(self) };

        unsafe { guard.weak.ptr.drop_in_place() };
    }

    /// Drops the value along with the `RawWeak` object, assuming the value pointed to by `ptr` is
    /// initialized,
    #[inline]
    pub(super) unsafe fn assume_init_drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
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
    /// - `self` should only be handled by the same `RcOps` implementation.
    #[inline]
    pub(crate) unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<R>(ptr: NonNull<()>)
        where
            R: RcOps,
        {
            if !is_dangling(ptr) {
                unsafe { increment_weak_ref_count::<R>(ptr) };
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
    /// - `self` should only be handled by the same `RcOps` implementation.
    /// - `self` is non-dangling.
    pub(crate) unsafe fn clone_unchecked<R>(&self) -> Self
    where
        A: Clone,
        R: RcOps,
    {
        unsafe {
            increment_weak_ref_count::<R>(self.ptr.cast());

            Self::from_raw_parts(self.ptr, self.alloc.clone())
        }
    }

    /// Drops this weak pointer.
    #[inline]
    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        if !is_dangling(self.ptr.cast()) {
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    /// Drops this weak pointer, assuming `self` is non-dangling.
    #[inline]
    pub(super) unsafe fn drop_unchecked<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        let is_last_weak_ref = unsafe { decrement_weak_ref_count::<R>(self.ptr.cast()) };

        if is_last_weak_ref {
            let rc_layout = unsafe { RcLayout::from_value_ptr_unchecked(self.ptr) };

            unsafe { super::deallocate::<A>(self.ptr.cast(), &self.alloc, rc_layout) }
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

    /// Returns the `RefCounts` object inside the reference-counted allocation if `self` is
    /// non-dangling.
    #[cfg(not(no_sync))]
    pub(crate) fn ref_counts(&self) -> Option<&RefCounts> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.ref_counts_unchecked() })
    }

    /// Returns the `RefCounts` object inside the reference-counted allocation, assume `self` is
    /// non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    pub(super) unsafe fn ref_counts_unchecked(&self) -> &RefCounts {
        unsafe { super::ref_counts_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
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
        unsafe { super::strong_count_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    /// Returns the weak reference count object inside the reference-counted allocation if `self`
    /// is non-dangling.
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
        unsafe { super::weak_count_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    /// Sets the contained pointer to a new value.
    ///
    /// # Safety
    ///
    /// - `ptr` should be a valid pointer to a value object that lives in a reference-counted
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
    /// `self` should only be handled by the same `RcOps` implementation.
    pub(crate) unsafe fn upgrade<R>(&self) -> Option<RawRc<T, A>>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<R>(value_ptr: NonNull<()>) -> bool
        where
            R: RcOps,
        {
            (!is_dangling(value_ptr))
                && unsafe { R::upgrade(super::strong_count_ptr_from_value_ptr(value_ptr).as_ref()) }
        }

        unsafe {
            inner::<R>(self.ptr.cast()).then(|| RawRc::from_raw_parts(self.ptr, self.alloc.clone()))
        }
    }
}

impl<T, A> RawWeak<T, A> {
    pub(crate) fn new_dangling() -> Self
    where
        A: Default,
    {
        Self::new_dangling_in(A::default())
    }

    pub(crate) const fn new_dangling_in(alloc: A) -> Self {
        unsafe { Self::from_raw_parts(NonNull::without_provenance(DANGLING_WEAK_ADDRESS), alloc) }
    }

    pub(crate) fn try_new_uninit<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        super::try_allocate_uninit::<A, STRONG_COUNT>(T::RC_LAYOUT)
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub(crate) fn try_new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        super::try_allocate_uninit_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub(crate) fn try_new_zeroed<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        super::try_allocate_zeroed::<A, STRONG_COUNT>(T::RC_LAYOUT)
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub(crate) fn try_new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        super::try_allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        let (ptr, alloc) = super::allocate_uninit::<A, STRONG_COUNT>(T::RC_LAYOUT);

        unsafe { Self::from_raw_parts(ptr.cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe {
            Self::from_raw_parts(
                super::allocate_uninit_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT).cast(),
                alloc,
            )
        }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        let (ptr, alloc) = super::allocate_zeroed::<A, STRONG_COUNT>(T::RC_LAYOUT);

        unsafe { Self::from_raw_parts(ptr.cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe {
            Self::from_raw_parts(
                super::allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT).cast(),
                alloc,
            )
        }
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
        R: RcOps,
    {
        unsafe {
            let result = self.ptr.read();

            self.drop_unchecked::<R>();

            result
        }
    }
}

impl<T, A> RawWeak<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    fn allocate<F>(length: usize, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(RcLayout) -> (NonNull<()>, A),
    {
        let rc_layout = RcLayout::new_array::<T>(length);
        let (ptr, alloc) = allocate_fn(rc_layout);

        unsafe { Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast(), length), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    fn allocate_in<F>(length: usize, alloc: A, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(&A, RcLayout) -> NonNull<()>,
    {
        let rc_layout = RcLayout::new_array::<T>(length);
        let ptr = allocate_fn(&alloc, rc_layout);

        unsafe { Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast(), length), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::allocate(length, super::allocate_uninit::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, super::allocate_uninit_in::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::allocate(length, super::allocate_zeroed::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, super::allocate_zeroed_in::<A, STRONG_COUNT>)
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
