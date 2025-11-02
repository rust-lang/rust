use core::alloc::{AllocError, Allocator};
use core::any::Any;
use core::cell::UnsafeCell;
#[cfg(not(no_global_oom_handling))]
use core::clone::CloneToUninit;
use core::error::{Error, Request};
use core::fmt::{self, Debug, Display, Formatter, Pointer};
use core::hash::{Hash, Hasher};
#[cfg(not(no_global_oom_handling))]
use core::iter::TrustedLen;
use core::marker::{PhantomData, Unsize};
#[cfg(not(no_global_oom_handling))]
use core::mem::{self, ManuallyDrop};
use core::mem::{DropGuard, MaybeUninit};
#[cfg(not(no_global_oom_handling))]
use core::ops::DerefMut;
use core::ops::{CoerceUnsized, DispatchFromDyn};
use core::pin::PinCoerceUnsized;
#[cfg(not(no_global_oom_handling))]
use core::ptr;
use core::ptr::NonNull;
#[cfg(not(no_global_oom_handling))]
use core::str;

use crate::alloc::Global;
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::MakeMutStrategy;
use crate::raw_rc::RefCounter;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::raw_unique_rc::RawUniqueRc;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::raw_weak;
use crate::raw_rc::raw_weak::RawWeak;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::rc_alloc;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::rc_layout::{RcLayout, RcLayoutExt};
use crate::raw_rc::rc_value_pointer::RcValuePointer;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

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
        unsafe { increment_strong_ref_count::<R>(RcValuePointer::from_value_ptr(ptr.cast())) };
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
            unsafe {
                R::from_raw_counter(value_ptr.weak_count_ptr().as_ref()).downgrade_increment_weak();
            }
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

        unsafe { inner::<R>(RcValuePointer::from_value_ptr(ptr)) }
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
        /// Returns a drop guard that sets the pointer in `rc` to `ptr` on drop.
        ///
        /// # Safety
        ///
        /// - `ptr` must point to a valid reference counted value that can be deallocated with the
        ///   allocator associated with `rc`.
        /// - The value pointed to by `ptr` must have an unowned strong reference count that can be
        ///   taken ownership by `rc`.
        unsafe fn set_rc_ptr_on_drop<'a, T, A>(
            rc: &'a mut RawRc<T, A>,
            ptr: NonNull<T>,
        ) -> impl DerefMut<Target = &'a mut RawRc<T, A>>
        where
            T: ?Sized,
        {
            DropGuard::new(rc, move |rc| unsafe { rc.weak.set_ptr(ptr) })
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

                        let guard = raw_weak::new_weak_guard::<T, A, R>(&mut self.weak);

                        let new_ptr = rc_alloc::allocate_with_bytes_in::<A, 1>(
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
                        let mut update_ptr_on_drop = set_rc_ptr_on_drop(self, new_ptr);

                        // `MakeMutStrategy::Move` guarantees that the strong count is zero, also we
                        // have copied the value to a new allocation, so we can pretend the original
                        // `RawRc` is now essentially an `RawWeak` object, we can call the `RawWeak`
                        // destructor to finish the cleanup.
                        update_ptr_on_drop.weak.drop_unchecked::<R>();
                    }
                    MakeMutStrategy::Clone => {
                        // There are multiple owners of the value, we need to clone the value into a
                        // new allocation.

                        let new_ptr = rc_alloc::allocate_with_in::<A, _, 1>(
                            &self.allocator(),
                            rc_layout,
                            |dst_ptr| {
                                T::clone_to_uninit(self.as_ref(), dst_ptr.as_ptr().as_ptr().cast())
                            },
                        );

                        let new_ptr = NonNull::from_raw_parts(
                            new_ptr.as_ptr(),
                            ptr::metadata(self.weak.as_ptr().as_ptr()),
                        );

                        // Ensure the value pointer in `self` is updated to `new_ptr`.
                        let mut update_ptr_on_drop = set_rc_ptr_on_drop(self, new_ptr);

                        // Manually drop old `RawRc`.
                        update_ptr_on_drop.drop::<R>();
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
    /// # Safety
    ///
    /// `weak` must be non-dangling.
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
        let (ptr, alloc) = rc_alloc::allocate_with::<A, _, 1>(T::RC_LAYOUT, |ptr| unsafe {
            ptr.as_ptr().cast().write(f())
        });

        unsafe { Self::from_raw_parts(ptr.as_ptr().cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    unsafe fn new_cyclic_impl<F, R>(mut weak: RawWeak<T, A>, data_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RefCounter,
    {
        let guard = unsafe { raw_weak::new_weak_guard::<T, A, R>(&mut weak) };
        let data = data_fn(&guard);

        mem::forget(guard);

        unsafe { RawUniqueRc::from_weak_with_value(weak, data).into_rc::<R>() }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn new_cyclic<F, R>(data_fn: F) -> Self
    where
        A: Allocator + Default,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RefCounter,
    {
        let weak = RawWeak::new_uninit::<0>();

        unsafe { Self::new_cyclic_impl::<F, R>(weak, data_fn) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn new_cyclic_in<F, R>(data_fn: F, alloc: A) -> Self
    where
        A: Allocator,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RefCounter,
    {
        let weak = RawWeak::new_uninit_in::<0>(alloc);

        unsafe { Self::new_cyclic_impl::<F, R>(weak, data_fn) }
    }

    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
    pub(crate) unsafe fn into_inner<R>(self) -> Option<T>
    where
        A: Allocator,
        R: RefCounter,
    {
        let is_last_strong_ref = unsafe { decrement_strong_ref_count::<R>(self.value_ptr()) };

        is_last_strong_ref.then(|| unsafe { self.weak.assume_init_into_inner::<R>() })
    }

    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
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
                R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()).try_lock_strong_count()
            }
        }

        let is_last_strong_ref = unsafe { inner::<R>(self.value_ptr()) };

        if is_last_strong_ref {
            Ok(unsafe { self.weak.assume_init_into_inner::<R>() })
        } else {
            Err(self)
        }
    }

    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
    pub(crate) unsafe fn unwrap_or_clone<R>(self) -> T
    where
        T: Clone,
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees `rc` will only be accessed with the same `RefCounter`
        // implementation.
        unsafe { self.try_unwrap::<R>() }.unwrap_or_else(|rc| {
            // SAFETY: Caller guarantees `rc` will only be accessed with the same `RefCounter`
            // implementation, and the `rc` local variable will not be accessed again after the
            // drop guard being triggered.
            let guard = DropGuard::new(rc, |mut rc| unsafe { rc.drop::<R>() });

            T::clone(guard.as_ref())
        })
    }
}

impl<T, A> RawRc<MaybeUninit<T>, A> {
    pub(crate) fn try_new_uninit() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_uninit_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_zeroed() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_zeroed::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_zeroed_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_zeroed_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_in::<1>(alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_in::<1>(alloc)) }
    }

    pub(crate) unsafe fn assume_init(self) -> RawRc<T, A> {
        unsafe { self.cast() }
    }
}

impl<T, A> RawRc<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    fn from_trusted_len_iter<I>(iter: I) -> Self
    where
        A: Allocator + Default,
        I: TrustedLen<Item = T>,
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
        ) -> impl DerefMut<Target = (NonNull<T>, NonNull<T>)> {
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

        let rc_layout = RcLayout::new_array::<T>(length);

        let (ptr, alloc) = rc_alloc::allocate_with::<A, _, 1>(rc_layout, |ptr| {
            let ptr = ptr.as_ptr().cast::<T>();
            let mut guard = unsafe { drop_range_on_drop::<T>(ptr, ptr) };

            // SAFETY: `iter` is `TrustedLen`, we can assume we will write correct number of
            // elements to the buffer.
            iter.for_each(|value| unsafe {
                guard.1.write(value);
                guard.1 = guard.1.add(1);
            });

            mem::forget(guard);
        });

        // SAFETY: We have written `length` of `T` values to the buffer, the buffer is now
        // initialized.
        unsafe {
            Self::from_raw_parts(
                NonNull::slice_from_raw_parts(ptr.as_ptr().cast::<T>(), length),
                alloc,
            )
        }
    }

    pub(crate) unsafe fn into_array<const N: usize, R>(self) -> Option<RawRc<[T; N], A>>
    where
        A: Allocator,
        R: RefCounter,
    {
        match RawRc::<[T; N], A>::try_from(self) {
            Ok(result) => Some(result),
            Err(mut raw_rc) => {
                unsafe { raw_rc.drop::<R>() };

                None
            }
        }
    }
}

impl<T, A> RawRc<[MaybeUninit<T>], A> {
    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice_in::<1>(length, alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice_in::<1>(length, alloc)) }
    }

    /// # Safety
    ///
    /// All `MaybeUninit<T>`s values contained by `self` must be initialized.
    pub(crate) unsafe fn assume_init(self) -> RawRc<[T], A> {
        unsafe { self.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

impl<A> RawRc<dyn Any, A> {
    pub(crate) fn downcast<T>(self) -> Result<RawRc<T, A>, Self>
    where
        T: Any,
    {
        if self.as_ref().is::<T>() { Ok(unsafe { self.downcast_unchecked() }) } else { Err(self) }
    }

    /// # Safety
    ///
    /// `self` must point to a valid `T` value.
    pub(crate) unsafe fn downcast_unchecked<T>(self) -> RawRc<T, A>
    where
        T: Any,
    {
        unsafe { self.cast() }
    }
}

impl<T, A> AsRef<T> for RawRc<T, A>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { self.weak.as_ptr().as_ref() }
    }
}

impl<T, U, A> CoerceUnsized<RawRc<U, A>> for RawRc<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Debug for RawRc<T, A>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Debug>::fmt(self.as_ref(), f)
    }
}

impl<T, A> Display for RawRc<T, A>
where
    T: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Display>::fmt(self.as_ref(), f)
    }
}

impl<T, U> DispatchFromDyn<RawRc<U, Global>> for RawRc<T, Global>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Error for RawRc<T, A>
where
    T: Error + ?Sized,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        T::source(self.as_ref())
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        T::cause(self.as_ref())
    }

    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        T::provide(self.as_ref(), request);
    }
}

impl<T, A> Pointer for RawRc<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <&T as Pointer>::fmt(&self.as_ref(), f)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> Default for RawRc<T, A>
where
    T: Default,
    A: Allocator + Default,
{
    fn default() -> Self {
        Self::new_with(T::default)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> Default for RawRc<[T], A>
where
    A: Allocator + Default,
{
    fn default() -> Self {
        RawRc::<[T; 0], A>::default()
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> Default for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn default() -> Self {
        let empty_slice = RawRc::<[u8], A>::default();

        // SAFETY: Empty slice is a valid `str`.
        unsafe { empty_slice.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as *mut _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<T> for RawRc<T, A>
where
    A: Allocator + Default,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<Box<T, A>> for RawRc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
    fn from(value: Box<T, A>) -> Self {
        let value_ref = &*value;
        let alloc_ref = Box::allocator(&value);

        unsafe {
            let value_ptr = rc_alloc::allocate_with_value_in::<T, A, 1>(value_ref, alloc_ref);
            let (box_ptr, alloc) = Box::into_raw_with_allocator(value);

            drop(Box::from_raw_in(box_ptr as *mut ManuallyDrop<T>, &alloc));

            Self::from_raw_parts(value_ptr, alloc)
        }
    }
}

#[cfg(not(no_global_oom_handling))]
trait SpecRawRcFromSlice<T> {
    fn spec_from_slice(slice: &[T]) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> SpecRawRcFromSlice<T> for RawRc<[T], A>
where
    T: Clone,
    A: Allocator + Default,
{
    default fn spec_from_slice(slice: &[T]) -> Self {
        Self::from_trusted_len_iter(slice.iter().cloned())
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> SpecRawRcFromSlice<T> for RawRc<[T], A>
where
    T: Copy,
    A: Allocator + Default,
{
    fn spec_from_slice(slice: &[T]) -> Self {
        let (ptr, alloc) = rc_alloc::allocate_with_value::<[T], A, 1>(slice);

        unsafe { Self::from_raw_parts(ptr, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<&[T]> for RawRc<[T], A>
where
    T: Clone,
    A: Allocator + Default,
{
    fn from(value: &[T]) -> Self {
        Self::spec_from_slice(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<&mut [T]> for RawRc<[T], A>
where
    T: Clone,
    A: Allocator + Default,
{
    fn from(value: &mut [T]) -> Self {
        Self::from(&*value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> From<&str> for RawRc<str, A>
where
    A: Allocator + Default,
{
    #[inline]
    fn from(value: &str) -> Self {
        let rc_of_bytes = RawRc::<[u8], A>::from(value.as_bytes());

        unsafe { rc_of_bytes.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> From<&mut str> for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn from(value: &mut str) -> Self {
        Self::from(&*value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl From<String> for RawRc<str, Global> {
    fn from(value: String) -> Self {
        let rc_of_bytes = RawRc::<[u8], Global>::from(value.into_bytes());

        unsafe { rc_of_bytes.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

impl<A> From<RawRc<str, A>> for RawRc<[u8], A> {
    fn from(value: RawRc<str, A>) -> Self {
        unsafe { value.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, const N: usize, A> From<[T; N]> for RawRc<[T], A>
where
    A: Allocator + Default,
{
    fn from(value: [T; N]) -> Self {
        RawRc::new(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<Vec<T, A>> for RawRc<[T], A>
where
    A: Allocator,
{
    fn from(value: Vec<T, A>) -> Self {
        let src = &*value;
        let alloc = value.allocator();
        let value_ptr = rc_alloc::allocate_with_value_in::<[T], A, 1>(src, alloc);
        let (vec_ptr, _length, capacity, alloc) = value.into_raw_parts_with_alloc();

        unsafe {
            drop(Vec::from_raw_parts_in(vec_ptr, 0, capacity, &alloc));

            Self::from_raw_parts(value_ptr, alloc)
        }
    }
}

impl<T, const N: usize, A> TryFrom<RawRc<[T], A>> for RawRc<[T; N], A> {
    type Error = RawRc<[T], A>;

    fn try_from(value: RawRc<[T], A>) -> Result<Self, Self::Error> {
        if value.as_ref().len() == N { Ok(unsafe { value.cast() }) } else { Err(value) }
    }
}

#[cfg(not(no_global_oom_handling))]
trait SpecRawRcFromIter<I> {
    fn spec_from_iter(iter: I) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<I> SpecRawRcFromIter<I> for RawRc<[I::Item], Global>
where
    I: Iterator,
{
    default fn spec_from_iter(iter: I) -> Self {
        Self::from(iter.collect::<Vec<_>>())
    }
}

#[cfg(not(no_global_oom_handling))]
impl<I> SpecRawRcFromIter<I> for RawRc<[I::Item], Global>
where
    I: TrustedLen,
{
    fn spec_from_iter(iter: I) -> Self {
        Self::from_trusted_len_iter(iter)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T> FromIterator<T> for RawRc<[T], Global> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::spec_from_iter(iter.into_iter())
    }
}

impl<T, A> Hash for RawRc<T, A>
where
    T: Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        T::hash(self.as_ref(), state);
    }
}

// Hack to allow specializing on `Eq` even though `Eq` has a method.
#[rustc_unsafe_specialization_marker]
trait MarkerEq: PartialEq<Self> {}

impl<T> MarkerEq for T where T: Eq {}

trait SpecPartialEq {
    fn spec_eq(&self, other: &Self) -> bool;
    fn spec_ne(&self, other: &Self) -> bool;
}

impl<T, A> SpecPartialEq for RawRc<T, A>
where
    T: PartialEq + ?Sized,
{
    #[inline]
    default fn spec_eq(&self, other: &Self) -> bool {
        T::eq(self.as_ref(), other.as_ref())
    }

    #[inline]
    default fn spec_ne(&self, other: &Self) -> bool {
        T::ne(self.as_ref(), other.as_ref())
    }
}

/// We're doing this specialization here, and not as a more general optimization on `&T`, because it
/// would otherwise add a cost to all equality checks on refs. We assume that `RawArc`s are used to
/// store large values, that are slow to clone, but also heavy to check for equality, causing this
/// cost to pay off more easily. It's also more likely to have two `RawArc` clones, that point to
/// the same value, than two `&T`s.
///
/// We can only do this when `T: Eq` as a `PartialEq` might be deliberately irreflexive.
impl<T, A> SpecPartialEq for RawRc<T, A>
where
    T: MarkerEq + ?Sized,
{
    #[inline]
    fn spec_eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other) || T::eq(self.as_ref(), other.as_ref())
    }

    #[inline]
    fn spec_ne(&self, other: &Self) -> bool {
        Self::ptr_ne(self, other) && T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialEq for RawRc<T, A>
where
    T: PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        Self::spec_eq(self, other)
    }

    fn ne(&self, other: &Self) -> bool {
        Self::spec_ne(self, other)
    }
}

impl<T, A> Eq for RawRc<T, A> where T: Eq + ?Sized {}

impl<T, A> PartialOrd for RawRc<T, A>
where
    T: PartialOrd + ?Sized,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        T::partial_cmp(self.as_ref(), other.as_ref())
    }

    fn lt(&self, other: &Self) -> bool {
        T::lt(self.as_ref(), other.as_ref())
    }

    fn le(&self, other: &Self) -> bool {
        T::le(self.as_ref(), other.as_ref())
    }

    fn gt(&self, other: &Self) -> bool {
        T::gt(self.as_ref(), other.as_ref())
    }

    fn ge(&self, other: &Self) -> bool {
        T::ge(self.as_ref(), other.as_ref())
    }
}

impl<T, A> Ord for RawRc<T, A>
where
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        T::cmp(self.as_ref(), other.as_ref())
    }
}

unsafe impl<T, A> PinCoerceUnsized for RawRc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
}
