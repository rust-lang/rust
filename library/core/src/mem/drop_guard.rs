use crate::fmt::{self, Debug};
use crate::marker::Destruct;
use crate::mem::ManuallyDrop;
use crate::ops::{Deref, DerefMut};

/// Wrap a value and run a closure when dropped.
///
/// This is useful for quickly creating destructors inline.
///
/// # Examples
///
/// ```rust
/// # #![allow(unused)]
/// #![feature(drop_guard)]
///
/// use std::mem::DropGuard;
///
/// {
///     // Create a new guard around a string that will
///     // print its value when dropped.
///     let s = String::from("Chashu likes tuna");
///     let mut s = DropGuard::with(s, |s| println!("{s}"));
///
///     // Modify the string contained in the guard.
///     s.push_str("!!!");
///
///     // The guard will be dropped here, printing:
///     // "Chashu likes tuna!!!"
/// }
/// ```
#[unstable(feature = "drop_guard", issue = "144426")]
#[doc(alias = "ScopeGuard")]
#[doc(alias = "defer")]
pub struct DropGuard<T, F>
where
    F: FnOnce(T),
{
    inner: ManuallyDrop<T>,
    f: ManuallyDrop<F>,
}

impl DropGuard<(), UnitFn> {
    /// Create a new instance of `DropGuard` with only a closure, no value.
    ///
    /// `DropGuard::new(|| ...)` is equivalent to `DropGuard::with((), |()| ...)`.
    ///
    /// # Example
    ///
    /// Enabling and then disabling a Unix terminal's [raw mode] within some
    /// block of code is a good use for `DropGuard`. Whether the block ends
    /// through successful completion, an unwinding panic, or early
    /// `return`/`break`/`continue`/`?`, the raw mode guard will ensure the
    /// disabling takes place.
    ///
    /// [raw mode]: https://man7.org/linux/man-pages/man3/termios.3.html#:~:text=Raw%20mode
    ///
    /// ```
    /// #![feature(drop_guard)]
    ///
    /// use std::mem::DropGuard;
    /// #
    /// # struct Terminal;
    /// # impl Terminal {
    /// #     fn enable_raw_mode(&self) {}
    /// #     fn disable_raw_mode(&self) {}
    /// # }
    /// # let terminal = Terminal;
    ///
    /// {
    ///     terminal.enable_raw_mode();
    ///     let _raw_mode_guard = DropGuard::new(|| terminal.disable_raw_mode());
    ///
    ///     // Write to terminal in raw mode. Upon end of this scope, raw mode ends.
    /// }
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[must_use]
    pub const fn new(f: impl FnOnce()) -> DropGuard<(), impl FnOnce(())> {
        DropGuard::with((), |()| f())
    }
}

impl<T, F> DropGuard<T, F>
where
    F: FnOnce(T),
{
    /// Create a new instance of `DropGuard` holding a value of type `T`.
    ///
    /// The value (`inner`) is provided to the closure that runs during drop,
    /// but also remains accessible to the surrounding code through the guard's
    /// `Deref`/`DerefMut`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// #![feature(drop_guard)]
    ///
    /// use std::mem::DropGuard;
    ///
    /// let value = String::from("Chashu likes tuna");
    /// let guard = DropGuard::with(value, |s| println!("{s}"));
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[must_use]
    pub const fn with(inner: T, f: F) -> Self {
        Self { inner: ManuallyDrop::new(inner), f: ManuallyDrop::new(f) }
    }

    /// Consumes the `DropGuard`, returning the wrapped value.
    ///
    /// This will not execute the closure. It is typically preferred to call
    /// this function instead of `mem::forget` because it will return the stored
    /// value and drop variables captured by the closure instead of leaking their
    /// owned resources.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// #![feature(drop_guard)]
    ///
    /// use std::mem::DropGuard;
    ///
    /// let value = String::from("Nori likes chicken");
    /// let guard = DropGuard::with(value, |s| println!("{s}"));
    /// assert_eq!(DropGuard::dismiss(guard), "Nori likes chicken");
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[rustc_const_unstable(feature = "const_drop_guard", issue = "none")]
    #[inline]
    pub const fn dismiss(guard: Self) -> T
    where
        F: [const] Destruct,
    {
        // First we ensure that dropping the guard will not trigger
        // its destructor
        let mut guard = ManuallyDrop::new(guard);

        // Next we manually read the stored value from the guard.
        //
        // SAFETY: this is safe because we've taken ownership of the guard.
        let value = unsafe { ManuallyDrop::take(&mut guard.inner) };

        // Finally we drop the stored closure. We do this *after* having read
        // the value, so that even if the closure's `drop` function panics,
        // unwinding still tries to drop the value.
        //
        // SAFETY: this is safe because we've taken ownership of the guard.
        unsafe { ManuallyDrop::drop(&mut guard.f) };
        value
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
const impl<T, F> Deref for DropGuard<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
const impl<T, F> DerefMut for DropGuard<T, F>
where
    F: FnOnce(T),
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
#[rustc_const_unstable(feature = "const_drop_guard", issue = "none")]
const impl<T, F> Drop for DropGuard<T, F>
where
    F: [const] FnOnce(T),
{
    fn drop(&mut self) {
        // SAFETY: `DropGuard` is in the process of being dropped.
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };

        // SAFETY: `DropGuard` is in the process of being dropped.
        let f = unsafe { ManuallyDrop::take(&mut self.f) };

        f(inner);
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
impl<T, F> Debug for DropGuard<T, F>
where
    T: Debug,
    F: FnOnce(T),
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// A private placeholder that prevents using turbofish in the `DropGuard::new`
/// call (`DropGuard::<(), ???>::new(...)`) with anything other than `_` as the
/// second type parameter.
///
/// Not publicly nameable outside libcore and not on track for stabilization.
#[unstable(feature = "drop_guard_unit_fn", issue = "none")]
#[allow(missing_debug_implementations)]
pub enum UnitFn {}

#[unstable(feature = "drop_guard_unit_fn", issue = "none")]
impl FnOnce<((),)> for UnitFn {
    type Output = ();

    extern "rust-call" fn call_once(self, _args: ((),)) -> Self::Output {
        match self {}
    }
}
