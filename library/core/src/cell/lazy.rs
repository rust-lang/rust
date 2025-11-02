use super::UnsafeCell;
use crate::hint::unreachable_unchecked;
use crate::ops::{Deref, DerefMut};
use crate::{fmt, mem};

enum State<T, F> {
    Uninit(F),
    Init(T),
    Poisoned,
}

/// A value which is initialized on the first access.
///
/// For a thread-safe version of this struct, see [`std::sync::LazyLock`].
///
/// [`std::sync::LazyLock`]: ../../std/sync/struct.LazyLock.html
///
/// # Poisoning
///
/// If the initialization closure passed to [`LazyCell::new`] panics, the cell will be poisoned.
/// Once the cell is poisoned, any threads that attempt to access this cell (via a dereference
/// or via an explicit call to [`force()`]) will panic.
///
/// This concept is similar to that of poisoning in the [`std::sync::poison`] module. A key
/// difference, however, is that poisoning in `LazyCell` is _unrecoverable_. All future accesses of
/// the cell from other threads will panic, whereas a type in [`std::sync::poison`] like
/// [`std::sync::poison::Mutex`] allows recovery via [`PoisonError::into_inner()`].
///
/// [`force()`]: LazyCell::force
/// [`std::sync::poison`]: ../../std/sync/poison/index.html
/// [`std::sync::poison::Mutex`]: ../../std/sync/poison/struct.Mutex.html
/// [`PoisonError::into_inner()`]: ../../std/sync/poison/struct.PoisonError.html#method.into_inner
///
/// # Examples
///
/// ```
/// use std::cell::LazyCell;
///
/// let lazy: LazyCell<i32> = LazyCell::new(|| {
///     println!("initializing");
///     92
/// });
/// println!("ready");
/// println!("{}", *lazy);
/// println!("{}", *lazy);
///
/// // Prints:
/// //   ready
/// //   initializing
/// //   92
/// //   92
/// ```
#[stable(feature = "lazy_cell", since = "1.80.0")]
pub struct LazyCell<T, F = fn() -> T> {
    state: UnsafeCell<State<T, F>>,
}

impl<T, F: FnOnce() -> T> LazyCell<T, F> {
    /// Creates a new lazy value with the given initializing function.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::LazyCell;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyCell::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// ```
    #[inline]
    #[stable(feature = "lazy_cell", since = "1.80.0")]
    #[rustc_const_stable(feature = "lazy_cell", since = "1.80.0")]
    pub const fn new(f: F) -> LazyCell<T, F> {
        LazyCell { state: UnsafeCell::new(State::Uninit(f)) }
    }

    /// Consumes this `LazyCell` returning the stored value.
    ///
    /// Returns `Ok(value)` if `Lazy` is initialized and `Err(f)` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the cell is poisoned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell_into_inner)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyCell::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// assert_eq!(LazyCell::into_inner(lazy).ok(), Some("HELLO, WORLD!".to_string()));
    /// ```
    #[unstable(feature = "lazy_cell_into_inner", issue = "125623")]
    #[rustc_const_unstable(feature = "lazy_cell_into_inner", issue = "125623")]
    pub const fn into_inner(this: Self) -> Result<T, F> {
        match this.state.into_inner() {
            State::Init(data) => Ok(data),
            State::Uninit(f) => Err(f),
            State::Poisoned => panic_poisoned(),
        }
    }

    /// Forces the evaluation of this lazy value and returns a reference to
    /// the result.
    ///
    /// This is equivalent to the `Deref` impl, but is explicit.
    ///
    /// # Panics
    ///
    /// If the initialization closure panics (the one that is passed to the [`new()`] method), the
    /// panic is propagated to the caller, and the cell becomes poisoned. This will cause all future
    /// accesses of the cell (via [`force()`] or a dereference) to panic.
    ///
    /// [`new()`]: LazyCell::new
    /// [`force()`]: LazyCell::force
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::LazyCell;
    ///
    /// let lazy = LazyCell::new(|| 92);
    ///
    /// assert_eq!(LazyCell::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[inline]
    #[stable(feature = "lazy_cell", since = "1.80.0")]
    pub fn force(this: &LazyCell<T, F>) -> &T {
        // SAFETY:
        // This invalidates any mutable references to the data. The resulting
        // reference lives either until the end of the borrow of `this` (in the
        // initialized case) or is invalidated in `really_init` (in the
        // uninitialized case; `really_init` will create and return a fresh reference).
        let state = unsafe { &*this.state.get() };
        match state {
            State::Init(data) => data,
            // SAFETY: The state is uninitialized.
            State::Uninit(_) => unsafe { LazyCell::really_init(this) },
            State::Poisoned => panic_poisoned(),
        }
    }

    /// Forces the evaluation of this lazy value and returns a mutable reference to
    /// the result.
    ///
    /// # Panics
    ///
    /// If the initialization closure panics (the one that is passed to the [`new()`] method), the
    /// panic is propagated to the caller, and the cell becomes poisoned. This will cause all future
    /// accesses of the cell (via [`force()`] or a dereference) to panic.
    ///
    /// [`new()`]: LazyCell::new
    /// [`force()`]: LazyCell::force
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_get)]
    /// use std::cell::LazyCell;
    ///
    /// let mut lazy = LazyCell::new(|| 92);
    ///
    /// let p = LazyCell::force_mut(&mut lazy);
    /// assert_eq!(*p, 92);
    /// *p = 44;
    /// assert_eq!(*lazy, 44);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_get", issue = "129333")]
    pub fn force_mut(this: &mut LazyCell<T, F>) -> &mut T {
        #[cold]
        /// # Safety
        /// May only be called when the state is `Uninit`.
        unsafe fn really_init_mut<T, F: FnOnce() -> T>(state: &mut State<T, F>) -> &mut T {
            // INVARIANT: Always valid, but the value may not be dropped.
            struct PoisonOnPanic<T, F>(*mut State<T, F>);
            impl<T, F> Drop for PoisonOnPanic<T, F> {
                #[inline]
                fn drop(&mut self) {
                    // SAFETY: Invariant states it is valid, and we don't drop the old value.
                    unsafe {
                        self.0.write(State::Poisoned);
                    }
                }
            }

            let State::Uninit(f) = state else {
                // `unreachable!()` here won't optimize out because the function is cold.
                // SAFETY: Precondition.
                unsafe { unreachable_unchecked() };
            };
            // SAFETY: We never drop the state after we read `f`, and we write a valid value back
            // in any case, panic or success. `f` can't access the `LazyCell` because it is mutably
            // borrowed.
            let f = unsafe { core::ptr::read(f) };
            // INVARIANT: Initiated from mutable reference, don't drop because we read it.
            let guard = PoisonOnPanic(state);
            let data = f();
            // SAFETY: `PoisonOnPanic` invariant, and we don't drop the old value.
            unsafe {
                core::ptr::write(guard.0, State::Init(data));
            }
            core::mem::forget(guard);
            let State::Init(data) = state else { unreachable!() };
            data
        }

        let state = this.state.get_mut();
        match state {
            State::Init(data) => data,
            // SAFETY: `state` is `Uninit`.
            State::Uninit(_) => unsafe { really_init_mut(state) },
            State::Poisoned => panic_poisoned(),
        }
    }

    /// # Safety
    /// May only be called when the state is `Uninit`.
    #[cold]
    unsafe fn really_init(this: &LazyCell<T, F>) -> &T {
        // SAFETY:
        // This function is only called when the state is uninitialized,
        // so no references to `state` can exist except for the reference
        // in `force`, which is invalidated here and not accessed again.
        let state = unsafe { &mut *this.state.get() };
        // Temporarily mark the state as poisoned. This prevents reentrant
        // accesses and correctly poisons the cell if the closure panicked.
        let State::Uninit(f) = mem::replace(state, State::Poisoned) else { unreachable!() };

        let data = f();

        // SAFETY:
        // If the closure accessed the cell through something like a reentrant
        // mutex, but caught the panic resulting from the state being poisoned,
        // the mutable borrow for `state` will be invalidated, so we need to
        // go through the `UnsafeCell` pointer here. The state can only be
        // poisoned at this point, so using `write` to skip the destructor
        // of `State` should help the optimizer.
        unsafe { this.state.get().write(State::Init(data)) };

        // SAFETY:
        // The previous references were invalidated by the `write` call above,
        // so do a new shared borrow of the state instead.
        let state = unsafe { &*this.state.get() };
        let State::Init(data) = state else { unreachable!() };
        data
    }
}

impl<T, F> LazyCell<T, F> {
    /// Returns a mutable reference to the value if initialized. Otherwise (if uninitialized or
    /// poisoned), returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_get)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let mut lazy = LazyCell::new(|| 92);
    ///
    /// assert_eq!(LazyCell::get_mut(&mut lazy), None);
    /// let _ = LazyCell::force(&lazy);
    /// *LazyCell::get_mut(&mut lazy).unwrap() = 44;
    /// assert_eq!(*lazy, 44);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_get", issue = "129333")]
    pub fn get_mut(this: &mut LazyCell<T, F>) -> Option<&mut T> {
        let state = this.state.get_mut();
        match state {
            State::Init(data) => Some(data),
            _ => None,
        }
    }

    /// Returns a reference to the value if initialized. Otherwise (if uninitialized or poisoned),
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_get)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let lazy = LazyCell::new(|| 92);
    ///
    /// assert_eq!(LazyCell::get(&lazy), None);
    /// let _ = LazyCell::force(&lazy);
    /// assert_eq!(LazyCell::get(&lazy), Some(&92));
    /// ```
    #[inline]
    #[unstable(feature = "lazy_get", issue = "129333")]
    pub fn get(this: &LazyCell<T, F>) -> Option<&T> {
        // SAFETY:
        // This is sound for the same reason as in `force`: once the state is
        // initialized, it will not be mutably accessed again, so this reference
        // will stay valid for the duration of the borrow to `self`.
        let state = unsafe { &*this.state.get() };
        match state {
            State::Init(data) => Some(data),
            _ => None,
        }
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T, F: FnOnce() -> T> Deref for LazyCell<T, F> {
    type Target = T;

    /// # Panics
    ///
    /// If the initialization closure panics (the one that is passed to the [`new()`] method), the
    /// panic is propagated to the caller, and the cell becomes poisoned. This will cause all future
    /// accesses of the cell (via [`force()`] or a dereference) to panic.
    ///
    /// [`new()`]: LazyCell::new
    /// [`force()`]: LazyCell::force
    #[inline]
    fn deref(&self) -> &T {
        LazyCell::force(self)
    }
}

#[stable(feature = "lazy_deref_mut", since = "1.89.0")]
impl<T, F: FnOnce() -> T> DerefMut for LazyCell<T, F> {
    /// # Panics
    ///
    /// If the initialization closure panics (the one that is passed to the [`new()`] method), the
    /// panic is propagated to the caller, and the cell becomes poisoned. This will cause all future
    /// accesses of the cell (via [`force()`] or a dereference) to panic.
    ///
    /// [`new()`]: LazyCell::new
    /// [`force()`]: LazyCell::force
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        LazyCell::force_mut(self)
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T: Default> Default for LazyCell<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    #[inline]
    fn default() -> LazyCell<T> {
        LazyCell::new(T::default)
    }
}

#[stable(feature = "lazy_cell", since = "1.80.0")]
impl<T: fmt::Debug, F> fmt::Debug for LazyCell<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_tuple("LazyCell");
        match LazyCell::get(self) {
            Some(data) => d.field(data),
            None => d.field(&format_args!("<uninit>")),
        };
        d.finish()
    }
}

#[cold]
#[inline(never)]
const fn panic_poisoned() -> ! {
    panic!("LazyCell instance has previously been poisoned")
}
