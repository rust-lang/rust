//! A custom LazyLock+Cow suitable for holding borrowed, owned or lazy data.

use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::sync::LazyLock;

enum MaybeLazyInner<T: 'static + ToOwned + ?Sized, F> {
    Lazy(LazyLock<T::Owned, F>),
    Cow(Cow<'static, T>),
}

/// A custom LazyLock+Cow suitable for holding borrowed, owned or lazy data.
///
/// Technically this structure has 3 states: borrowed, owned and lazy
/// They can all be constructed from the [`MaybeLazy::borrowed`], [`MaybeLazy::owned`] and
/// [`MaybeLazy::lazy`] methods.
#[repr(transparent)]
pub struct MaybeLazy<T: 'static + ToOwned + ?Sized, F = fn() -> <T as ToOwned>::Owned> {
    // Inner state.
    //
    // Not to be inlined since we may want in the future to
    // make this struct usable to statics and we might need to
    // workaround const-eval limitation (particulary around drop).
    inner: MaybeLazyInner<T, F>,
}

impl<T: 'static + ?Sized + ToOwned, F: FnOnce() -> T::Owned> MaybeLazy<T, F> {
    /// Create a [`MaybeLazy`] from an borrowed `T`.
    #[inline]
    pub const fn borrowed(a: &'static T) -> Self {
        MaybeLazy { inner: MaybeLazyInner::Cow(Cow::Borrowed(a)) }
    }

    /// Create a [`MaybeLazy`] from an borrowed `T`.
    #[inline]
    pub const fn owned(a: T::Owned) -> Self {
        MaybeLazy { inner: MaybeLazyInner::Cow(Cow::Owned(a)) }
    }

    /// Create a [`MaybeLazy`] from a function-able `F`.
    #[inline]
    pub const fn lazied(f: F) -> Self {
        MaybeLazy { inner: MaybeLazyInner::Lazy(LazyLock::new(f)) }
    }
}

impl<T: 'static + ?Sized + ToOwned> MaybeLazy<T> {
    /// Create a [`MaybeLazy`] from a function pointer.
    #[inline]
    pub const fn lazy(a: fn() -> T::Owned) -> Self {
        Self::lazied(a)
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Clone>, F: FnOnce() -> T::Owned> Clone
    for MaybeLazy<T, F>
{
    #[inline]
    fn clone(&self) -> Self {
        MaybeLazy {
            inner: MaybeLazyInner::Cow(match &self.inner {
                MaybeLazyInner::Lazy(f) => Cow::Owned((*f).to_owned()),
                MaybeLazyInner::Cow(c) => c.clone(),
            }),
        }
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Default>, F: FnOnce() -> T::Owned> Default
    for MaybeLazy<T, F>
{
    #[inline]
    fn default() -> MaybeLazy<T, F> {
        MaybeLazy::owned(T::Owned::default())
    }
}

// `Debug`, `Display` and other traits below are implemented in terms of this `Deref`
impl<T: 'static + ?Sized + ToOwned<Owned: Borrow<T>>, F: FnOnce() -> T::Owned> Deref
    for MaybeLazy<T, F>
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        match &self.inner {
            MaybeLazyInner::Lazy(f) => (&**f).borrow(),
            MaybeLazyInner::Cow(c) => &*c,
        }
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Debug> + Debug, F: FnOnce() -> T::Owned> Debug
    for MaybeLazy<T, F>
{
    #[inline]
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, fmt)
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Display> + Display, F: FnOnce() -> T::Owned> Display
    for MaybeLazy<T, F>
{
    #[inline]
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&**self, fmt)
    }
}

impl<T: 'static + ?Sized + ToOwned, F: FnOnce() -> T::Owned> AsRef<T> for MaybeLazy<T, F> {
    #[inline]
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<
    T1: ?Sized + PartialEq<T2> + ToOwned,
    T2: ?Sized + ToOwned,
    F1: FnOnce() -> T1::Owned,
    F2: FnOnce() -> T2::Owned,
> PartialEq<MaybeLazy<T2, F2>> for MaybeLazy<T1, F1>
{
    #[inline]
    fn eq(&self, other: &MaybeLazy<T2, F2>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<F: FnOnce() -> String> PartialEq<&str> for MaybeLazy<str, F> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        &**self == *other
    }
}

impl<F: FnOnce() -> String> From<&'static str> for MaybeLazy<str, F> {
    #[inline]
    fn from(s: &'static str) -> MaybeLazy<str, F> {
        MaybeLazy::borrowed(s)
    }
}
