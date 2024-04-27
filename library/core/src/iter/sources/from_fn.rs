use crate::fmt;

/// Creates a new iterator where each iteration calls the provided closure
/// `F: FnMut() -> Option<T>`.
///
/// This allows creating a custom iterator with any behavior
/// without using the more verbose syntax of creating a dedicated type
/// and implementing the [`Iterator`] trait for it.
///
/// Note that the `FromFn` iterator doesn’t make assumptions about the behavior of the closure,
/// and therefore conservatively does not implement [`FusedIterator`],
/// or override [`Iterator::size_hint()`] from its default `(0, None)`.
///
/// The closure can use captures and its environment to track state across iterations. Depending on
/// how the iterator is used, this may require specifying the [`move`] keyword on the closure.
///
/// [`move`]: ../../std/keyword.move.html
/// [`FusedIterator`]: crate::iter::FusedIterator
///
/// # Examples
///
/// Let’s re-implement the counter iterator from [module-level documentation]:
///
/// [module-level documentation]: crate::iter
///
/// ```
/// let mut count = 0;
/// let counter = std::iter::from_fn(move || {
///     // Increment our count. This is why we started at zero.
///     count += 1;
///
///     // Check to see if we've finished counting or not.
///     if count < 6 {
///         Some(count)
///     } else {
///         None
///     }
/// });
/// assert_eq!(counter.collect::<Vec<_>>(), &[1, 2, 3, 4, 5]);
/// ```
#[inline]
#[stable(feature = "iter_from_fn", since = "1.34.0")]
pub fn from_fn<T, F>(f: F) -> FromFn<F>
where
    F: FnMut() -> Option<T>,
{
    FromFn(f)
}

/// An iterator where each iteration calls the provided closure `F: FnMut() -> Option<T>`.
///
/// This `struct` is created by the [`iter::from_fn()`] function.
/// See its documentation for more.
///
/// [`iter::from_fn()`]: from_fn
#[derive(Clone)]
#[stable(feature = "iter_from_fn", since = "1.34.0")]
pub struct FromFn<F>(F);

#[stable(feature = "iter_from_fn", since = "1.34.0")]
impl<T, F> Iterator for FromFn<F>
where
    F: FnMut() -> Option<T>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.0)()
    }
}

#[stable(feature = "iter_from_fn", since = "1.34.0")]
impl<F> fmt::Debug for FromFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FromFn").finish()
    }
}
