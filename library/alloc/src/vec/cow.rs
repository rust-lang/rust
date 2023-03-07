use crate::borrow::Cow;
use core::iter::FromIterator;

use super::Vec;

#[stable(feature = "cow_from_vec", since = "1.8.0")]
impl<'a, T: Clone> From<&'a [T]> for Cow<'a, [T]> {
    /// Creates a [`Borrowed`] variant of [`Cow`]
    /// from a slice.
    ///
    /// This conversion does not allocate or clone the data.
    ///
    /// [`Borrowed`]: crate::borrow::Cow::Borrowed
    fn from(s: &'a [T]) -> Cow<'a, [T]> {
        Cow::Borrowed(s)
    }
}

#[stable(feature = "cow_from_vec", since = "1.8.0")]
impl<'a, T: Clone> From<Vec<T>> for Cow<'a, [T]> {
    /// Creates an [`Owned`] variant of [`Cow`]
    /// from an owned instance of [`Vec`].
    ///
    /// This conversion does not allocate or clone the data.
    ///
    /// [`Owned`]: crate::borrow::Cow::Owned
    fn from(v: Vec<T>) -> Cow<'a, [T]> {
        Cow::Owned(v)
    }
}

#[stable(feature = "cow_from_vec_ref", since = "1.28.0")]
impl<'a, T: Clone> From<&'a Vec<T>> for Cow<'a, [T]> {
    /// Creates a [`Borrowed`] variant of [`Cow`]
    /// from a reference to [`Vec`].
    ///
    /// This conversion does not allocate or clone the data.
    ///
    /// [`Borrowed`]: crate::borrow::Cow::Borrowed
    fn from(v: &'a Vec<T>) -> Cow<'a, [T]> {
        Cow::Borrowed(v.as_slice())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> FromIterator<T> for Cow<'a, [T]>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Cow<'a, [T]> {
        Cow::Owned(FromIterator::from_iter(it))
    }
}
