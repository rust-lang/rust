use crate::fmt;
use crate::ops::{Coroutine, CoroutineState};
use crate::pin::Pin;

/// Creates a new iterator where each iteration calls the provided coroutine.
///
/// Similar to [`iter::from_fn`].
///
/// [`iter::from_fn`]: crate::iter::from_fn
///
/// # Examples
///
/// ```
/// #![feature(coroutines)]
/// #![feature(iter_from_coroutine)]
///
/// let it = std::iter::from_coroutine(#[coroutine] || {
///     yield 1;
///     yield 2;
///     yield 3;
/// });
/// let v: Vec<_> = it.collect();
/// assert_eq!(v, [1, 2, 3]);
/// ```
#[inline]
#[unstable(feature = "iter_from_coroutine", issue = "43122", reason = "coroutines are unstable")]
pub fn from_coroutine<G: Coroutine<Return = ()> + Unpin>(coroutine: G) -> FromCoroutine<G> {
    FromCoroutine(coroutine)
}

/// An iterator over the values yielded by an underlying coroutine.
///
/// This `struct` is created by the [`iter::from_coroutine()`] function. See its documentation for
/// more.
///
/// [`iter::from_coroutine()`]: from_coroutine
#[unstable(feature = "iter_from_coroutine", issue = "43122", reason = "coroutines are unstable")]
#[derive(Clone)]
pub struct FromCoroutine<G>(G);

#[unstable(feature = "iter_from_coroutine", issue = "43122", reason = "coroutines are unstable")]
impl<G: Coroutine<Return = ()> + Unpin> Iterator for FromCoroutine<G> {
    type Item = G::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        match Pin::new(&mut self.0).resume(()) {
            CoroutineState::Yielded(n) => Some(n),
            CoroutineState::Complete(()) => None,
        }
    }
}

#[unstable(feature = "iter_from_coroutine", issue = "43122", reason = "coroutines are unstable")]
impl<G> fmt::Debug for FromCoroutine<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FromCoroutine").finish()
    }
}
