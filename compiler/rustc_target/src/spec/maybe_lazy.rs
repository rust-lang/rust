//! A custom LazyLock+Cow suitable for holding borrowed, owned or lazy data.

use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::sync::LazyLock;

enum MaybeLazyInner<T: 'static + ToOwned + ?Sized> {
    Lazy(LazyLock<T::Owned>),
    Cow(Cow<'static, T>),
}

/// A custom LazyLock+Cow suitable for holding borrowed, owned or lazy data.
///
/// Technically this structure has 3 states: borrowed, owned and lazy
/// They can all be constructed from the [`MaybeLazy::borrowed`], [`MaybeLazy::owned`] and
/// [`MaybeLazy::lazy`] methods.
#[repr(transparent)]
pub struct MaybeLazy<T: 'static + ToOwned + ?Sized> {
    // Inner state.
    //
    // Not to be inlined since we may want in the future to
    // make this struct usable to statics and we might need to
    // workaround const-eval limitation (particulary around drop).
    inner: MaybeLazyInner<T>,
}

impl<T: 'static + ?Sized + ToOwned> MaybeLazy<T> {
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

    /// Create a [`MaybeLazy`] that is lazy by taking a function pointer.
    ///
    /// This function pointer cannot *ever* take a closure. User can potentially
    /// workaround that by using closure-to-fnptr or `const` items.
    #[inline]
    pub const fn lazy(a: fn() -> T::Owned) -> Self {
        MaybeLazy { inner: MaybeLazyInner::Lazy(LazyLock::new(a)) }
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Clone>> Clone for MaybeLazy<T> {
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

impl<T: 'static + ?Sized + ToOwned<Owned: Default>> Default for MaybeLazy<T> {
    #[inline]
    fn default() -> MaybeLazy<T> {
        MaybeLazy::lazy(T::Owned::default)
    }
}

// `Debug`, `Display` and other traits below are implemented in terms of this `Deref`
impl<T: 'static + ?Sized + ToOwned<Owned: Borrow<T>>> Deref for MaybeLazy<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        match &self.inner {
            MaybeLazyInner::Lazy(f) => (&**f).borrow(),
            MaybeLazyInner::Cow(c) => &*c,
        }
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Debug> + Debug> Debug for MaybeLazy<T> {
    #[inline]
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, fmt)
    }
}

impl<T: 'static + ?Sized + ToOwned<Owned: Display> + Display> Display for MaybeLazy<T> {
    #[inline]
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&**self, fmt)
    }
}

impl<T: 'static + ?Sized + ToOwned> AsRef<T> for MaybeLazy<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<B: ?Sized + PartialEq<C> + ToOwned, C: ?Sized + ToOwned> PartialEq<MaybeLazy<C>>
    for MaybeLazy<B>
{
    #[inline]
    fn eq(&self, other: &MaybeLazy<C>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl PartialEq<&str> for MaybeLazy<str> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        &**self == *other
    }
}

impl From<&'static str> for MaybeLazy<str> {
    #[inline]
    fn from(s: &'static str) -> MaybeLazy<str> {
        MaybeLazy::borrowed(s)
    }
}
