use crate::fmt;
use crate::iter::{FusedIterator, TrustedLen};
use crate::marker;

/// Creates an iterator that yields nothing.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // this could have been an iterator over i32, but alas, it's just not.
/// let mut nope = iter::empty::<i32>();
///
/// assert_eq!(None, nope.next());
/// ```
#[stable(feature = "iter_empty", since = "1.2.0")]
#[rustc_const_stable(feature = "const_iter_empty", since = "1.32.0")]
pub const fn empty<T>() -> Empty<T> {
    Empty(marker::PhantomData)
}

/// An iterator that yields nothing.
///
/// This `struct` is created by the [`empty()`] function. See its documentation for more.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iter_empty", since = "1.2.0")]
pub struct Empty<T>(marker::PhantomData<T>);

#[stable(feature = "iter_empty_send_sync", since = "1.42.0")]
unsafe impl<T> Send for Empty<T> {}
#[stable(feature = "iter_empty_send_sync", since = "1.42.0")]
unsafe impl<T> Sync for Empty<T> {}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T> fmt::Debug for Empty<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Empty").finish()
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Iterator for Empty<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> DoubleEndedIterator for Empty<T> {
    fn next_back(&mut self) -> Option<T> {
        None
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> ExactSizeIterator for Empty<T> {
    fn len(&self) -> usize {
        0
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Empty<T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Empty<T> {}

// not #[derive] because that adds a Clone bound on T,
// which isn't necessary.
#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Clone for Empty<T> {
    fn clone(&self) -> Empty<T> {
        Empty(marker::PhantomData)
    }
}

// not #[derive] because that adds a Default bound on T,
// which isn't necessary.
#[stable(feature = "iter_empty", since = "1.2.0")]
#[rustc_const_unstable(feature = "const_default_impls", issue = "87864")]
impl<T> const Default for Empty<T> {
    fn default() -> Empty<T> {
        Empty(marker::PhantomData)
    }
}
