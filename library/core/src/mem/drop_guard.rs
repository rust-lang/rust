use crate::fmt::{self, Debug};
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
///     let mut s = DropGuard::new(s, |s| println!("{s}"));
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

impl<T, F> DropGuard<T, F>
where
    F: FnOnce(T),
{
    /// Create a new instance of `DropGuard`.
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
    /// let guard = DropGuard::new(value, |s| println!("{s}"));
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[must_use]
    pub const fn new(inner: T, f: F) -> Self {
        Self { inner: ManuallyDrop::new(inner), f: ManuallyDrop::new(f) }
    }

    /// Consumes the `DropGuard`, returning the wrapped value.
    ///
    /// This will not execute the closure. This is implemented as an associated
    /// function to prevent any potential conflicts with any other methods called
    /// `into_inner` from the `Deref` and `DerefMut` impls.
    ///
    /// It is typically preferred to call this function instead of `mem::forget`
    /// because it will return the stored value and drop variables captured
    /// by the closure instead of leaking their owned resources.
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
    /// let guard = DropGuard::new(value, |s| println!("{s}"));
    /// assert_eq!(DropGuard::into_inner(guard), "Nori likes chicken");
    /// ```
    #[unstable(feature = "drop_guard", issue = "144426")]
    #[inline]
    pub fn into_inner(guard: Self) -> T {
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
impl<T, F> Deref for DropGuard<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &T {
        &*self.inner
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
impl<T, F> DerefMut for DropGuard<T, F>
where
    F: FnOnce(T),
{
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.inner
    }
}

#[unstable(feature = "drop_guard", issue = "144426")]
impl<T, F> Drop for DropGuard<T, F>
where
    F: FnOnce(T),
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
