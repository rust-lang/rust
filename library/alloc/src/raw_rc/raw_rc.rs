use core::alloc::{AllocError, Allocator};
use core::any::Any;
use core::cell::UnsafeCell;
use core::clone::CloneToUninit;
#[cfg(not(no_global_oom_handling))]
use core::iter::TrustedLen;
use core::marker::PhantomData;
#[cfg(not(no_global_oom_handling))]
use core::mem::SizedTypeProperties;
use core::mem::{DropGuard, MaybeUninit};
#[cfg(not(no_global_oom_handling))]
use core::ops::{ControlFlow, DerefMut, Try};
use core::ptr::NonNull;

#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::MakeMutStrategy;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::raw_weak;
use crate::raw_rc::raw_weak::RawWeak;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::rc_layout::{RcLayout, RcLayoutExt};
use crate::raw_rc::rc_value_pointer::RcValuePointer;
use crate::raw_rc::{RefCounter, rc_alloc};

/// Base implementation of a strong pointer. `RawRc` does not implement `Drop`; the user should call
/// `RawRc::drop` manually to destroy this object.
#[repr(transparent)]
pub(crate) struct RawRc<T, A>
where
    T: ?Sized,
{
    /// A `RawRc` is just a non-dangling `RawWeak` that has a strong reference count owned by the
    /// `RawRc` object. The weak pointer is always non-dangling.
    weak: RawWeak<T, A>,

    // Defines the ownership of `T` for drop-check.
    _phantom_data: PhantomData<T>,
}

impl<T, A> RawRc<T, A>
where
    T: ?Sized,
{
    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn clone_from_ref_in(value: &T, alloc: A) -> Self
    where
        A: Allocator,
        T: CloneToUninit,
    {
        let ptr = rc_alloc::allocate_with_cloned_in::<T, A, 1>(value, &alloc);

        unsafe { Self::from_raw_parts(ptr, alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn clone_from_ref(value: &T) -> Self
    where
        A: Allocator + Default,
        T: CloneToUninit,
    {
        let (ptr, alloc) = rc_alloc::allocate_with_cloned::<T, A, 1>(value);

        unsafe { Self::from_raw_parts(ptr, alloc) }
    }

    pub(crate) fn try_clone_from_ref_in(value: &T, alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
        T: CloneToUninit,
    {
        rc_alloc::try_allocate_with_cloned_in::<T, A, 1>(value, &alloc)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr, alloc) })
    }

    pub(crate) fn try_clone_from_ref(value: &T) -> Result<Self, AllocError>
    where
        A: Allocator + Default,
        T: CloneToUninit,
    {
        rc_alloc::try_allocate_with_cloned::<T, A, 1>(value)
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr, alloc) })
    }

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

    pub(crate) const fn as_ptr(&self) -> NonNull<T> {
        self.weak.as_ptr()
    }

    const fn as_ref(&self) -> &T {
        unsafe { self.as_ptr().as_ref() }
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

    pub(crate) unsafe fn decrement_strong_count<R>(ptr: NonNull<T>)
    where
        A: Allocator + Default,
        R: RefCounter,
    {
        unsafe { Self::decrement_strong_count_in::<R>(ptr, A::default()) };
    }

    pub(crate) unsafe fn decrement_strong_count_in<R>(ptr: NonNull<T>, alloc: A)
    where
        A: Allocator,
        R: RefCounter,
    {
        unsafe { RawRc::from_raw_parts(ptr, alloc).drop::<R>() };
    }

    pub(crate) unsafe fn increment_strong_count<R>(ptr: NonNull<T>)
    where
        R: RefCounter,
    {
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
    pub(crate) unsafe fn is_unique<R>(&self) -> bool
    where
        R: RefCounter,
    {
        unsafe { is_unique::<R>(self.value_ptr()) }
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
        /// - `ptr` must point to a valid reference-counted value that can be deallocated with the
        ///   allocator associated with `rc`.
        /// - The value pointed to by `ptr` must have an unowned strong reference count that can be
        ///   taken ownership of by `rc`.
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
                match strategy {
                    MakeMutStrategy::Move => {
                        // `R::make_mut` has set the strong reference count to zero, so the `RawRc`
                        // is essentially a `RawWeak` object whose value is initialized. This means
                        // we are the only owner of the value and can safely move it into a new
                        // allocation.

                        // `guard` ensures the old `RawRc` object is dropped even if the allocation
                        // panics.
                        let guard = raw_weak::new_weak_guard::<T, A, R>(&mut self.weak);

                        let new_ptr = rc_alloc::allocate_with_value_in_unchecked::<T, A, 1>(
                            guard.as_ptr().as_ref(),
                            &guard.allocator(),
                        );

                        // No panic occurred, defuse the guard.
                        DropGuard::dismiss(guard);

                        // Ensure the value pointer in `self` is updated to `new_ptr`.
                        let mut update_ptr_on_drop = set_rc_ptr_on_drop(self, new_ptr);

                        // `MakeMutStrategy::Move` guarantees that the strong count is zero, also we
                        // have copied the value to a new allocation, so we can pretend the original
                        // `RawRc` is now essentially an `RawWeak` object, we can call the `RawWeak`
                        // destructor to finish the cleanup.
                        update_ptr_on_drop.weak.drop_unchecked::<R>();
                    }
                    MakeMutStrategy::Clone => {
                        // There are multiple owners of the value, so we need to clone the value
                        // into a new allocation.

                        let new_ptr = rc_alloc::allocate_with_cloned_in::<T, A, 1>(
                            self.as_ref(),
                            self.allocator(),
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
    pub(crate) fn try_new_in(value: T, alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc)
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    #[inline]
    pub(crate) fn try_new(value: T) -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>()
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
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
    #[inline]
    pub(crate) fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<1>(), value) }
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

    /// Attempts to map the value in an `Rc`, reusing the allocation if possible.
    ///
    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn try_map<R, U>(
        mut self,
        f: impl FnOnce(&T) -> U,
    ) -> ControlFlow<U::Residual, RawRc<U::Output, A>>
    where
        A: Allocator,
        R: RefCounter,
        U: Try,
    {
        let result = if T::LAYOUT == U::Output::LAYOUT && unsafe { self.is_unique::<R>() } {
            let value = unsafe { self.as_ptr().read() };
            let mut allocation = unsafe { self.cast::<MaybeUninit<U::Output>>() };

            // Destruct `self` as `RawRc<MaybeUninit<U::Output>, A>` if `f` panics or returns a
            // failure value.
            let guard = unsafe { new_rc_guard::<MaybeUninit<U::Output>, A, R>(&mut allocation) };

            let mapped_value = f(&value).branch()?;

            drop(value);
            DropGuard::dismiss(guard);

            unsafe {
                allocation.get_mut_unchecked().write(mapped_value);

                allocation.assume_init()
            }
        } else {
            // Destruct `self` if `f` panics or returns a failure value.
            let guard = unsafe { new_rc_guard::<T, A, R>(&mut self) };

            let mapped_value = f(guard.as_ref()).branch()?;

            drop(guard);

            let alloc = self.into_raw_parts().1;

            RawRc::new_in(mapped_value, alloc)
        };

        ControlFlow::Continue(result)
    }

    /// Maps the value in an `RawRc`, reusing the allocation if possible.
    ///
    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn map<R, U>(self, f: impl FnOnce(&T) -> U) -> RawRc<U, A>
    where
        A: Allocator,
        R: RefCounter,
    {
        fn wrap_fn<T, U>(f: impl FnOnce(&T) -> U) -> impl FnOnce(&T) -> ControlFlow<!, U> {
            |x| ControlFlow::Continue(f(x))
        }

        let f = wrap_fn(f);

        match unsafe { self.try_map::<R, _>(f) } {
            ControlFlow::Continue(output) => output,
        }
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
        unsafe { self.try_unwrap::<R>() }.unwrap_or_else(|mut rc| {
            let guard = unsafe { new_rc_guard::<T, A, R>(&mut rc) };

            T::clone(guard.as_ref())
        })
    }
}

impl<T, A> RawRc<MaybeUninit<T>, A> {
    pub(crate) fn try_new_uninit_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_uninit() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_zeroed_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_zeroed_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_zeroed() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_zeroed::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_in::<1>(alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_in::<1>(alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed::<1>()) }
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

            DropGuard::dismiss(guard);
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

    fn try_into_array<const N: usize>(self) -> Result<RawRc<[T; N], A>, Self> {
        if self.as_ref().len() == N { Ok(unsafe { self.cast() }) } else { Err(self) }
    }

    pub(crate) unsafe fn into_array<const N: usize, R>(self) -> Option<RawRc<[T; N], A>>
    where
        A: Allocator,
        R: RefCounter,
    {
        match self.try_into_array::<N>() {
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
    pub(crate) fn new_uninit_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice_in::<1>(length, alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice_in::<1>(length, alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice::<1>(length)) }
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

/// Returns a drop guard that calls `Rc::drop::<R>()` on drop.
unsafe fn new_rc_guard<'a, T, A, R>(
    rc: &'a mut RawRc<T, A>,
) -> DropGuard<&'a mut RawRc<T, A>, impl FnOnce(&'a mut RawRc<T, A>)>
where
    T: ?Sized,
    A: Allocator,
    R: RefCounter,
{
    DropGuard::new(rc, |rc| unsafe { rc.drop::<R>() })
}
