use crate::ops::Deref;
use crate::{fmt, mem};

use super::UnsafeCell;

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
/// # Examples
///
/// ```
/// #![feature(lazy_cell)]
///
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
#[unstable(feature = "lazy_cell", issue = "109736")]
pub struct LazyCell<T, F = fn() -> T> {
    state: UnsafeCell<State<T, F>>,
}

impl<T, F: FnOnce() -> T> LazyCell<T, F> {
    /// Creates a new lazy value with the given initializing function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let hello = "Hello, World!".to_string();
    ///
    /// let lazy = LazyCell::new(|| hello.to_uppercase());
    ///
    /// assert_eq!(&*lazy, "HELLO, WORLD!");
    /// ```
    #[inline]
    #[unstable(feature = "lazy_cell", issue = "109736")]
    pub const fn new(f: F) -> LazyCell<T, F> {
        LazyCell { state: UnsafeCell::new(State::Uninit(f)) }
    }

    /// Consumes this `LazyCell` returning the stored value.
    ///
    /// Returns `Ok(value)` if `Lazy` is initialized and `Err(f)` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell)]
    /// #![feature(lazy_cell_consume)]
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
    #[unstable(feature = "lazy_cell_consume", issue = "109736")]
    pub fn into_inner(this: Self) -> Result<T, F> {
        match this.state.into_inner() {
            State::Init(data) => Ok(data),
            State::Uninit(f) => Err(f),
            State::Poisoned => panic!("LazyCell instance has previously been poisoned"),
        }
    }

    /// Forces the evaluation of this lazy value and returns a reference to
    /// the result.
    ///
    /// This is equivalent to the `Deref` impl, but is explicit.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(lazy_cell)]
    ///
    /// use std::cell::LazyCell;
    ///
    /// let lazy = LazyCell::new(|| 92);
    ///
    /// assert_eq!(LazyCell::force(&lazy), &92);
    /// assert_eq!(&*lazy, &92);
    /// ```
    #[inline]
    #[unstable(feature = "lazy_cell", issue = "109736")]
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
            State::Poisoned => panic!("LazyCell has previously been poisoned"),
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
    #[inline]
    fn get(&self) -> Option<&T> {
        // SAFETY:
        // This is sound for the same reason as in `force`: once the state is
        // initialized, it will not be mutably accessed again, so this reference
        // will stay valid for the duration of the borrow to `self`.
        let state = unsafe { &*self.state.get() };
        match state {
            State::Init(data) => Some(data),
            _ => None,
        }
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T, F: FnOnce() -> T> Deref for LazyCell<T, F> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        LazyCell::force(self)
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: Default> Default for LazyCell<T> {
    /// Creates a new lazy value using `Default` as the initializing function.
    #[inline]
    fn default() -> LazyCell<T> {
        LazyCell::new(T::default)
    }
}

#[unstable(feature = "lazy_cell", issue = "109736")]
impl<T: fmt::Debug, F> fmt::Debug for LazyCell<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_tuple("LazyCell");
        match self.get() {
            Some(data) => d.field(data),
            None => d.field(&format_args!("<uninit>")),
        };
        d.finish()
    }
}
