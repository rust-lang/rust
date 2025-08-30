use core::alloc::{AllocError, Allocator};
use core::cell::UnsafeCell;
#[cfg(not(no_global_oom_handling))]
use core::clone::CloneToUninit;
#[cfg(not(no_global_oom_handling))]
use core::marker::PhantomData;
#[cfg(not(no_global_oom_handling))]
use core::mem;
use core::ptr::NonNull;

use crate::raw_rc::RefCounter;
use crate::raw_rc::raw_weak::RawWeak;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::rc_layout::{RcLayout, RcLayoutExt};
use crate::raw_rc::rc_value_pointer::RcValuePointer;

/// Decrements strong reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
#[inline]
unsafe fn decrement_strong_ref_count<R>(value_ptr: RcValuePointer) -> bool
where
    R: RefCounter,
{
    unsafe { R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()).decrement() }
}

/// Increments strong reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
#[inline]
unsafe fn increment_strong_ref_count<R>(value_ptr: RcValuePointer)
where
    R: RefCounter,
{
    unsafe { R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()).increment() };
}

#[inline]
unsafe fn is_unique<R>(value_ptr: RcValuePointer) -> bool
where
    R: RefCounter,
{
    let ref_counts = unsafe { value_ptr.ref_counts_ptr().as_ref() };

    unsafe {
        R::is_unique(R::from_raw_counter(&ref_counts.strong), R::from_raw_counter(&ref_counts.weak))
    }
}

/// Base implementation of a strong pointer. `RawRc` does not implement `Drop`, user should call
/// `RawRc::drop` manually to drop this object.
#[repr(transparent)]
pub(crate) struct RawRc<T, A>
where
    T: ?Sized,
{
    /// A `RawRc` is just a non-dangling `RawWeak` that has a strong reference count that is owned
    /// by the `RawRc` object. The weak pointer is always non-dangling.
    weak: RawWeak<T, A>,

    // Defines the ownership of `T` for drop-check.
    _phantom_data: PhantomData<T>,
}

impl<T, A> RawRc<T, A>
where
    T: ?Sized,
{
    /// # Safety
    ///
    /// - `ptr` points to a value inside a reference-counted allocation.
    /// - The allocation can be freed by `A::default()`.
    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        A: Default,
    {
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    /// # Safety
    ///
    /// - `ptr` points to a value inside a reference-counted allocation.
    /// - The allocation can be freed by `alloc`.
    pub(crate) unsafe fn from_raw_parts(ptr: NonNull<T>, alloc: A) -> Self {
        unsafe { Self::from_weak(RawWeak::from_raw_parts(ptr, alloc)) }
    }

    /// # Safety
    ///
    /// `weak` must have at least one unowned strong reference count. The newly created `RawRc` will
    /// take the ownership of exactly one strong reference count.
    pub(super) unsafe fn from_weak(weak: RawWeak<T, A>) -> Self {
        Self { weak, _phantom_data: PhantomData }
    }

    pub(crate) fn allocator(&self) -> &A {
        &self.weak.allocator()
    }

    pub(crate) fn as_ptr(&self) -> NonNull<T> {
        self.weak.as_ptr()
    }

    pub(crate) unsafe fn cast<U>(self) -> RawRc<U, A> {
        unsafe { RawRc::from_weak(self.weak.cast()) }
    }

    #[inline]
    pub(crate) unsafe fn cast_with<U, F>(self, f: F) -> RawRc<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        unsafe { RawRc::from_weak(self.weak.cast_with(f)) }
    }

    #[inline]
    pub(crate) unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RefCounter,
    {
        unsafe {
            increment_strong_ref_count::<R>(self.value_ptr());

            Self::from_raw_parts(self.weak.as_ptr(), self.allocator().clone())
        }
    }

    pub(crate) unsafe fn decrement_strong_count<R: RefCounter>(ptr: NonNull<T>)
    where
        A: Allocator + Default,
    {
        unsafe { Self::decrement_strong_count_in::<R>(ptr, A::default()) };
    }

    pub(crate) unsafe fn decrement_strong_count_in<R: RefCounter>(ptr: NonNull<T>, alloc: A)
    where
        A: Allocator,
    {
        unsafe { RawRc::from_raw_parts(ptr, alloc).drop::<R>() };
    }

    pub(crate) unsafe fn increment_strong_count<R: RefCounter>(ptr: NonNull<T>) {
        unsafe { increment_strong_ref_count::<R>(RcValuePointer::new(ptr.cast())) };
    }

    pub(crate) unsafe fn downgrade<R>(&self) -> RawWeak<T, A>
    where
        A: Clone,
        R: RefCounter,
    {
        unsafe fn inner<R>(value_ptr: RcValuePointer)
        where
            R: RefCounter,
        {
            unsafe { R::from_raw_counter(value_ptr.weak_count_ptr().as_ref()).downgrade() };
        }

        unsafe {
            inner::<R>(self.value_ptr());

            RawWeak::from_raw_parts(self.weak.as_ptr(), self.allocator().clone())
        }
    }

    #[inline]
    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        let is_last_strong_ref = unsafe { decrement_strong_ref_count::<R>(self.value_ptr()) };

        if is_last_strong_ref {
            unsafe { self.weak.assume_init_drop::<R>() }
        }
    }

    pub(crate) unsafe fn get_mut<R>(&mut self) -> Option<&mut T>
    where
        R: RefCounter,
    {
        unsafe fn inner<R>(value_ptr: RcValuePointer) -> Option<RcValuePointer>
        where
            R: RefCounter,
        {
            unsafe { is_unique::<R>(value_ptr) }.then_some(value_ptr)
        }

        let (ptr, metadata) = self.weak.as_ptr().to_raw_parts();

        unsafe { inner::<R>(RcValuePointer::new(ptr)) }
            .map(|ptr| unsafe { NonNull::from_raw_parts(ptr.as_ptr(), metadata).as_mut() })
    }

    /// Returns a mutable reference to the contained value.
    ///
    /// # Safety
    ///
    /// No other active references to the contained value should exist, and no new references to the
    /// contained value will be acquired for the duration of the returned borrow.
    pub(crate) unsafe fn get_mut_unchecked(&mut self) -> &mut T {
        // SAFETY: The caller guarantees that we can access the contained value exclusively. Note
        // that we can't create mutable references that have access to reference counters, because
        // the caller only guarantee exclusive access to the contained value, not the reference
        // counters.
        unsafe { self.weak.as_ptr().as_mut() }
    }

    pub(crate) fn into_raw(self) -> NonNull<T> {
        self.weak.into_raw()
    }

    pub(crate) fn into_raw_parts(self) -> (NonNull<T>, A) {
        self.weak.into_raw_parts()
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn make_mut<R>(&mut self) -> &mut T
    where
        T: CloneToUninit,
        A: Allocator + Clone,
        R: RefCounter,
    {
        use core::ptr;

        use crate::raw_rc::MakeMutStrategy;
        use crate::raw_rc::raw_weak::WeakGuard;

        struct SetRcPtrOnDrop<'a, T, A>
        where
            T: ?Sized,
        {
            rc: &'a mut RawRc<T, A>,
            new_ptr: NonNull<T>,
        }

        impl<T, A> Drop for SetRcPtrOnDrop<'_, T, A>
        where
            T: ?Sized,
        {
            fn drop(&mut self) {
                unsafe { self.rc.weak.set_ptr(self.new_ptr) };
            }
        }

        unsafe {
            let ref_counts = self.ref_counts();

            if let Some(strategy) = R::make_mut(
                R::from_raw_counter(&ref_counts.strong),
                R::from_raw_counter(&ref_counts.weak),
            ) {
                let rc_layout = RcLayout::from_value_ptr_unchecked(self.weak.as_ptr());

                match strategy {
                    MakeMutStrategy::Move => {
                        // `R::make_mut` has made strong reference count to zero, so the `RawRc`
                        // object is essentially a `RawWeak` object but has its value initialized.
                        // This means we are the only owner of the value and we can safely move the
                        // value into a new allocation.

                        // This guarantees to drop old `RawRc` object even if the allocation
                        // panics.
                        let guard = WeakGuard::<T, A, R>::new(&mut self.weak);

                        let new_ptr = super::allocate_with_bytes_in::<A, 1>(
                            guard.as_ptr().cast(),
                            &guard.allocator(),
                            rc_layout,
                        );

                        // No panic happens, defuse the guard.
                        mem::forget(guard);

                        let new_ptr = NonNull::from_raw_parts(
                            new_ptr.as_ptr(),
                            ptr::metadata(self.weak.as_ptr().as_ptr()),
                        );

                        // Ensure the value pointer in `self` is updated to `new_ptr`.
                        let update_ptr_on_drop = SetRcPtrOnDrop { rc: self, new_ptr };

                        // `MakeMutStrategy::Move` guarantees that the strong count is zero, also we
                        // have copied the value to a new allocation, so we can pretend the original
                        // `RawRc` is now essentially an `RawWeak` object, we can call the `RawWeak`
                        // destructor to finish the cleanup.
                        update_ptr_on_drop.rc.weak.drop_unchecked::<R>();
                    }
                    MakeMutStrategy::Clone => {
                        // There are multiple owners of the value, we need to clone the value into a
                        // new allocation.

                        let new_ptr = super::allocate_with_in::<A, _, 1>(
                            &self.allocator(),
                            rc_layout,
                            |dst_ptr| {
                                T::clone_to_uninit(
                                    self.as_ptr().as_ref(),
                                    dst_ptr.as_ptr().as_ptr().cast(),
                                )
                            },
                        );

                        let new_ptr = NonNull::from_raw_parts(
                            new_ptr.as_ptr(),
                            ptr::metadata(self.weak.as_ptr().as_ptr()),
                        );

                        // Ensure the value pointer in `self` is updated to `new_ptr`.
                        let update_ptr_on_drop = SetRcPtrOnDrop { rc: self, new_ptr };

                        // Manually drop old `RawRc`.
                        update_ptr_on_drop.rc.drop::<R>();
                    }
                }
            }

            self.get_mut_unchecked()
        }
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        RawWeak::ptr_eq(&self.weak, &other.weak)
    }

    pub(crate) fn ptr_ne(&self, other: &Self) -> bool {
        RawWeak::ptr_ne(&self.weak, &other.weak)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn ref_counts(&self) -> &crate::raw_rc::RefCounts {
        unsafe { self.weak.ref_counts_unchecked() }
    }

    pub(crate) fn strong_count(&self) -> &UnsafeCell<usize> {
        unsafe { self.weak.strong_count_unchecked() }
    }

    pub(crate) fn weak_count(&self) -> &UnsafeCell<usize> {
        unsafe { self.weak.weak_count_unchecked() }
    }

    #[inline]
    fn value_ptr(&self) -> RcValuePointer {
        // SAFETY: `self.weak` is guaranteed to be non-dangling.
        unsafe { self.weak.value_ptr_unchecked() }
    }
}

impl<T, A> RawRc<T, A> {
    unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        unsafe {
            weak.as_ptr().write(value);

            Self::from_weak(weak)
        }
    }

    #[inline]
    pub(crate) fn try_new(value: T) -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>()
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    #[inline]
    pub(crate) fn try_new_in(value: T, alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc)
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<1>(), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit_in::<1>(alloc), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    fn new_with<F>(f: F) -> Self
    where
        A: Allocator + Default,
        F: FnOnce() -> T,
    {
        let (ptr, alloc) = super::allocate_with::<A, _, 1>(T::RC_LAYOUT, |ptr| unsafe {
            ptr.as_ptr().cast().write(f())
        });

        unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) }
    }

    pub(crate) unsafe fn into_inner<R>(self) -> Option<T>
    where
        A: Allocator,
        R: RefCounter,
    {
        let is_last_strong_ref = unsafe { decrement_strong_ref_count::<R>(self.value_ptr()) };

        is_last_strong_ref.then(|| unsafe { self.weak.assume_init_into_inner::<R>() })
    }

    pub(crate) unsafe fn try_unwrap<R>(self) -> Result<T, RawRc<T, A>>
    where
        A: Allocator,
        R: RefCounter,
    {
        unsafe fn inner<R>(value_ptr: RcValuePointer) -> bool
        where
            R: RefCounter,
        {
            unsafe {
                R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()).lock_strong_count()
            }
        }

        let is_last_strong_ref = unsafe { inner::<R>(self.value_ptr()) };

        if is_last_strong_ref {
            Ok(unsafe { self.weak.assume_init_into_inner::<R>() })
        } else {
            Err(self)
        }
    }

    pub(crate) unsafe fn unwrap_or_clone<R>(self) -> T
    where
        T: Clone,
        A: Allocator,
        R: RefCounter,
    {
        /// Calls `RawRc::drop` on drop.
        struct Guard<'a, T, A, R>
        where
            T: ?Sized,
            A: Allocator,
            R: RefCounter,
        {
            rc: &'a mut RawRc<T, A>,
            _phantom_data: PhantomData<R>,
        }

        impl<T, A, R> Drop for Guard<'_, T, A, R>
        where
            T: ?Sized,
            A: Allocator,
            R: RefCounter,
        {
            fn drop(&mut self) {
                unsafe { self.rc.drop::<R>() };
            }
        }

        unsafe {
            self.try_unwrap::<R>().unwrap_or_else(|mut rc| {
                let guard = Guard::<T, A, R> { rc: &mut rc, _phantom_data: PhantomData };

                T::clone(guard.rc.as_ptr().as_ref())
            })
        }
    }
}
