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
        let state = unsafe { &*this.state.get() };
        match state {
            State::Init(data) => data,
            State::Uninit(_) => unsafe { LazyCell::really_init(this) },
            State::Poisoned => panic!("LazyCell has previously been poisoned"),
        }
    }

    /// # Safety
    /// May only be called when the state is `Uninit`.
    #[cold]
    unsafe fn really_init(this: &LazyCell<T, F>) -> &T {
        let state = unsafe { &mut *this.state.get() };
        // Temporarily mark the state as poisoned. This prevents reentrant
        // accesses and correctly poisons the cell if the closure panicked.
        let State::Uninit(f) = mem::replace(state, State::Poisoned) else { unreachable!() };

        let data = f();

        // If the closure accessed the cell, the mutable borrow will be
        // invalidated, so create a new one here.
        let state = unsafe { &mut *this.state.get() };
        *state = State::Init(data);

        // A reference obtained by downcasting from the mutable borrow
        // would become stale if other references are created in `force`.
        // Borrow the state directly instead.
        let state = unsafe { &*this.state.get() };
        let State::Init(data) = state else { unreachable!() };
        data
    }
}

impl<T, F> LazyCell<T, F> {
    #[inline]
    fn get(&self) -> Option<&T> {
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
