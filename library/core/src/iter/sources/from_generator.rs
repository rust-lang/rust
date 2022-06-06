use crate::ops::{Generator, GeneratorState};
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
pub fn from_generator<G: Generator<Return = ()> + Unpin>(
    generator: G,
) -> impl Iterator<Item = G::Yield> {
    FromGenerator(generator)
}

struct FromGenerator<G>(G);

impl<G: Generator<Return = ()> + Unpin> Iterator for FromGenerator<G> {
    type Item = G::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        match Pin::new(&mut self.0).resume(()) {
            GeneratorState::Yielded(n) => Some(n),
            GeneratorState::Complete(()) => None,
        }
    }
}
