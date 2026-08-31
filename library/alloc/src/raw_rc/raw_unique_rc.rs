use core::alloc::{Allocator, AllocatorClone};
use core::marker::PhantomData;
#[cfg(not(no_global_oom_handling))]
use core::mem::{DropGuard, SizedTypeProperties};
#[cfg(not(no_global_oom_handling))]
use core::ops::{ControlFlow, Try};

use crate::raw_rc::RefCounter;
use crate::raw_rc::raw_rc::RawRc;
#[cfg(not(no_global_oom_handling))]
use crate::raw_rc::raw_weak;
use crate::raw_rc::raw_weak::RawWeak;
use crate::raw_rc::rc_value_pointer::RcValuePointer;

/// A uniquely owned `RawRc` that allows multiple weak references but only one strong reference.
/// `RawUniqueRc` does not implement `Drop`; the user should call `RawUniqueRc::drop` manually to
/// destroy this object.
#[repr(transparent)]
pub(crate) struct RawUniqueRc<T, A>
where
    T: ?Sized,
{
    // A `RawUniqueRc` is just a non-dangling `RawWeak` that has zero strong count but with the
    // value initialized.
    weak: RawWeak<T, A>,

    // Defines the ownership of `T` for drop-check.
    _marker: PhantomData<T>,

    // Invariance is necessary for soundness: once other `RawWeak` references exist, we already
    // have a form of shared mutability!
    _marker2: PhantomData<*mut T>,
}

impl<T, A> RawUniqueRc<T, A>
where
    T: ?Sized,
{
    /// Increments the weak count and returns the corresponding `RawWeak` object.
    ///
    /// # Safety
    ///
    /// - `self`, the derived `RawWeak`s or `RawRc`s must be handled only by the same `RefCounter`
    ///   implementation.
    pub(crate) unsafe fn downgrade<R>(&self) -> RawWeak<T, A>
    where
        A: AllocatorClone,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees consistency of `R` and `self.weak` is guaranteed to be
        // non-dangling.
        unsafe { self.weak.clone_unchecked::<R>() }
    }

    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees no access to the allocation will happen through `self`, and
        // the contained value is guaranteed to be initialized in `RawUniqueRc`.
        unsafe { self.weak.assume_init_drop::<R>() };
    }

    pub(crate) unsafe fn into_rc<R>(self) -> RawRc<T, A>
    where
        R: RefCounter,
    {
        unsafe fn inner<R>(value_ptr: RcValuePointer)
        where
            R: RefCounter,
        {
            // SAFETY: Caller guarantees the validify of `value_ptr` and the consistency of `R`.
            unsafe { R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()) }
                .unlock_strong_count();
        }

        // SAFETY: Caller guarantees the consistency of `R`
        unsafe {
            inner::<R>(self.weak.value_ptr_unchecked());

            RawRc::from_weak(self.weak)
        }
    }
}

impl<T, A> RawUniqueRc<T, A> {
    #[cfg(not(no_global_oom_handling))]
    pub(super) unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        // SAFETY: Caller guarantees we have exclusive access to the value storage.
        unsafe { weak.as_ptr().write(value) };

        Self { weak, _marker: PhantomData, _marker2: PhantomData }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        let weak = RawWeak::new_uninit_in::<0>(alloc);

        // SAFETY: we have exclusive access to the allocation thus the value storage.
        unsafe { Self::from_weak_with_value(weak, value) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        let weak = RawWeak::new_uninit::<0>();

        // SAFETY: we have exclusive access to the allocation thus the value storage.
        unsafe { Self::from_weak_with_value(weak, value) }
    }

    /// Attempts to map the value in a `RawUniqueRc`, reusing the allocation if possible.
    ///
    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn try_map<R, U>(
        mut self,
        f: impl FnOnce(T) -> U,
    ) -> ControlFlow<U::Residual, RawUniqueRc<U::Output, A>>
    where
        A: Allocator,
        R: RefCounter,
        U: Try,
    {
        // Destruct `self` as a `RawWeak<T, A>` if `f` panics or returns a failure value.
        //
        // SAFETY: Caller guarantees the consistency of `R`.
        let guard = unsafe { raw_weak::new_weak_guard::<T, A, R>(&mut self.weak) };

        let (allocation, mapped_value) = if T::LAYOUT == U::Output::LAYOUT
            // SAFETY: Caller guarantees the consistency of `R`.
            && R::unique_rc_weak_count(unsafe { R::from_raw_counter(guard.weak_count_unchecked()) })
                == 1
        {
            // SAFETY: `guard.as_ptr()` is guaranteed to point to a properly initialized value. And
            // we will not access it after `read` no matter `f` succeeds, fails or panics.
            let mapped_value = f(unsafe { guard.as_ptr().read() }).branch()?;

            // Avoid deallocation on success, reuse the allocation.
            DropGuard::dismiss(guard);

            // SAFETY: We have checked the compatibility of `T` and `U::Output`.
            let allocation = unsafe { self.weak.cast() };

            (allocation, mapped_value)
        } else {
            // SAFETY: `guard.as_ptr()` is guaranteed to point to a properly initialized value. And
            // we will not access it after `read` no matter `f` succeeds, fails or panics.
            let value = unsafe { guard.as_ptr().read() };

            drop(guard);

            let mapped_value = f(value).branch()?;
            let allocation = RawWeak::new_uninit_in::<0>(self.weak.into_raw_parts().1);

            (allocation, mapped_value)
        };

        // SAFETY: Both branch guarantees exclusive ownership of the allocation.
        ControlFlow::Continue(unsafe {
            RawUniqueRc::from_weak_with_value(allocation, mapped_value)
        })
    }

    /// Maps the value in a `RawUniqueRc`, reusing the allocation if possible.
    ///
    /// # Safety
    ///
    /// All accesses to `self` must use the same `RefCounter` implementation for `R`.
    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn map<R, U>(self, f: impl FnOnce(T) -> U) -> RawUniqueRc<U, A>
    where
        A: Allocator,
        R: RefCounter,
    {
        fn wrap_fn<T, U>(f: impl FnOnce(T) -> U) -> impl FnOnce(T) -> ControlFlow<!, U> {
            |x| ControlFlow::Continue(f(x))
        }

        let f = wrap_fn(f);

        // SAFETY: Caller guarantees the consistency of `R`.
        match unsafe { self.try_map::<R, _>(f) } {
            ControlFlow::Continue(output) => output,
        }
    }
}
