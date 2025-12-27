use core::alloc::Allocator;
use core::fmt::{self, Debug, Display, Formatter, Pointer};
use core::hash::{Hash, Hasher};
use core::marker::{PhantomData, Unsize};
#[cfg(not(no_global_oom_handling))]
use core::mem::{DropGuard, SizedTypeProperties};
use core::ops::{CoerceUnsized, DispatchFromDyn};
#[cfg(not(no_global_oom_handling))]
use core::ops::{ControlFlow, Try};

use crate::alloc::Global;
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
        A: Clone,
        R: RefCounter,
    {
        // SAFETY: Caller guarantees we only use the same `Rc` implementation and `self.weak` is
        // never dangling.
        unsafe { self.weak.clone_unchecked::<R>() }
    }

    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RefCounter,
    {
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
            unsafe { R::from_raw_counter(value_ptr.strong_count_ptr().as_ref()) }
                .unlock_strong_count();
        }

        unsafe {
            inner::<R>(self.weak.value_ptr_unchecked());

            RawRc::from_weak(self.weak)
        }
    }
}

impl<T, A> RawUniqueRc<T, A> {
    #[cfg(not(no_global_oom_handling))]
    pub(super) unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        unsafe { weak.as_ptr().write(value) };

        Self { weak, _marker: PhantomData, _marker2: PhantomData }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit_in::<0>(alloc), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<0>(), value) }
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
        let guard = unsafe { raw_weak::new_weak_guard::<T, A, R>(&mut self.weak) };

        let (allocation, mapped_value) = if T::LAYOUT == U::Output::LAYOUT
            && R::unique_rc_weak_count(unsafe { R::from_raw_counter(guard.weak_count_unchecked()) })
                == 1
        {
            let mapped_value = f(unsafe { guard.as_ptr().read() }).branch()?;

            // Avoid deallocation on success, reuse the allocation.
            DropGuard::dismiss(guard);

            let allocation = unsafe { self.weak.cast() };

            (allocation, mapped_value)
        } else {
            let value = unsafe { guard.as_ptr().read() };

            drop(guard);

            let mapped_value = f(value).branch()?;
            let allocation = RawWeak::new_uninit_in::<0>(self.weak.into_raw_parts().1);

            (allocation, mapped_value)
        };

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

        match unsafe { self.try_map::<R, _>(f) } {
            ControlFlow::Continue(output) => output,
        }
    }
}

impl<T, A> AsMut<T> for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.weak.as_ptr().as_mut() }
    }
}

impl<T, A> AsRef<T> for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { self.weak.as_ptr().as_ref() }
    }
}

impl<T, U, A> CoerceUnsized<RawUniqueRc<U, A>> for RawUniqueRc<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
    A: Allocator,
{
}

impl<T, A> Debug for RawUniqueRc<T, A>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Debug>::fmt(self.as_ref(), f)
    }
}

impl<T, U> DispatchFromDyn<RawUniqueRc<U, Global>> for RawUniqueRc<T, Global>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Display for RawUniqueRc<T, A>
where
    T: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Display>::fmt(self.as_ref(), f)
    }
}

impl<T, A> Eq for RawUniqueRc<T, A> where T: Eq + ?Sized {}

impl<T, A> Hash for RawUniqueRc<T, A>
where
    T: Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        T::hash(self.as_ref(), state);
    }
}

impl<T, A> Ord for RawUniqueRc<T, A>
where
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        T::cmp(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialEq for RawUniqueRc<T, A>
where
    T: PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        T::eq(self.as_ref(), other.as_ref())
    }

    fn ne(&self, other: &Self) -> bool {
        T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialOrd for RawUniqueRc<T, A>
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

impl<T, A> Pointer for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <&T as Pointer>::fmt(&self.as_ref(), f)
    }
}
