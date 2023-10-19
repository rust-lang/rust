use crate::fmt;
use crate::ops::{Coroutine, CoroutineState};
use crate::pin::Pin;

/// Creates a new iterator where each iteration calls the provided generator.
///
/// Similar to [`iter::from_fn`].
///
/// [`iter::from_fn`]: crate::iter::from_fn
///
/// # Examples
///
/// ```
/// #![feature(generators)]
/// #![feature(iter_from_generator)]
///
/// let it = std::iter::from_generator(|| {
///     yield 1;
///     yield 2;
///     yield 3;
/// });
/// let v: Vec<_> = it.collect();
/// assert_eq!(v, [1, 2, 3]);
/// ```
#[inline]
#[unstable(feature = "iter_from_generator", issue = "43122", reason = "generators are unstable")]
pub fn from_generator<G: Coroutine<Return = ()> + Unpin>(generator: G) -> FromCoroutine<G> {
    FromCoroutine(generator)
}

/// An iterator over the values yielded by an underlying generator.
///
/// This `struct` is created by the [`iter::from_generator()`] function. See its documentation for
/// more.
///
/// [`iter::from_generator()`]: from_generator
#[unstable(feature = "iter_from_generator", issue = "43122", reason = "generators are unstable")]
#[derive(Clone)]
pub struct FromCoroutine<G>(G);

#[unstable(feature = "iter_from_generator", issue = "43122", reason = "generators are unstable")]
impl<G: Coroutine<Return = ()> + Unpin> Iterator for FromCoroutine<G> {
    type Item = G::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        match Pin::new(&mut self.0).resume(()) {
            CoroutineState::Yielded(n) => Some(n),
            CoroutineState::Complete(()) => None,
        }
    }
}

#[unstable(feature = "iter_from_generator", issue = "43122", reason = "generators are unstable")]
impl<G> fmt::Debug for FromCoroutine<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FromCoroutine").finish()
    }
}
