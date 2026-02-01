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
/// use std::mem::defer;
///
/// {
///     // Create a new guard that will do something
///     // when dropped.
///     defer! {
///         println!("Goodbye, world!");
///     }
///
///     // The guard will be dropped here, printing:
///     // "Goodbye, world!"
/// }
///
/// {
///     // Create a new guard that will do something
///     // when dropped.
///     let _guard = DropGuard::new(|| println!("Goodbye, world!"));
///
///     // The guard will be dropped here, printing:
///     // "Goodbye, world!"
/// }
///
/// {
///     // Create a new guard around a string that will
///     // print its value when dropped.
///     let s = String::from("Chashu likes tuna");
///     let mut s = DropGuard::with_value(s, |s| println!("{s}"));
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
pub struct DropGuard<T = (), F = UnitFn<fn()>>
where
    F: FnOnce(T),
{
    inner: ManuallyDrop<T>,
    f: ManuallyDrop<F>,
}

/// Create an anonymous `DropGuard` with a cleanup closure.
///
/// The macro takes statements, which are the body of a closure
/// that will run when the scope is exited.
///
/// # Example
///
/// ```rust
/// # #![allow(unused)]
/// #![feature(drop_guard)]
///
/// use std::mem::defer;
///
/// defer! {
///     println!("Goodbye, world!");
/// }
/// ```
#[unstable(feature = "drop_guard", issue = "144426")]
pub macro defer($($t:tt)*) {
    let _guard = $crate::mem::DropGuard::new(|| { $($t)* });
}

impl<F> DropGuard<(), UnitFn<F>>
where
    F: FnOnce(),
{
    /// Create a new instance of `DropGuard` with a cleanup closure.
    ///
    /// # Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// #![feature(drop_guard)]
    ///
    /// use std::mem::DropGuard;
    ///
    /// let guard = DropGuard::new(|| println!("Goodbye, world!"));
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[must_use]
    pub const fn new(f: F) -> Self {
        Self { inner: ManuallyDrop::new(()), f: ManuallyDrop::new(UnitFn(f)) }
    }
}

impl<T, F> DropGuard<T, F>
where
    F: FnOnce(T),
{
    /// Create a new instance of `DropGuard` with a value and a cleanup closure.
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
    /// let guard = DropGuard::with_value(value, |s| println!("{s}"));
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[must_use]
    pub const fn with_value(inner: T, f: F) -> Self {
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
    /// let guard = DropGuard::with_value(value, |s| println!("{s}"));
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
impl<T, F> const Deref for DropGuard<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &T {
        &*self.inner
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T, F> const DerefMut for DropGuard<T, F>
where
    F: FnOnce(T),
{
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.inner
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
#[rustc_const_unstable(feature = "const_drop_guard", issue = "none")]
impl<T, F> const Drop for DropGuard<T, F>
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

#[unstable(feature = "drop_guard", issue = "144426")]
pub struct UnitFn<F>(F);

#[unstable(feature = "drop_guard", issue = "144426")]
impl<F> FnOnce<((),)> for UnitFn<F>
where
    F: FnOnce(),
{
    type Output = ();

    extern "rust-call" fn call_once(self, _args: ((),)) -> Self::Output {
        (self.0)()
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
impl<F> Debug for UnitFn<F>
where
    F: FnOnce(),
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnitFn")
    }
}
