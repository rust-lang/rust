use crate::fmt;
use crate::iter::FusedIterator;

/// Creates a new iterator where each successive item is computed based on the preceding one.
///
/// The iterator starts with the given first item (if any)
/// and calls the given `FnMut(&T) -> Option<T>` closure to compute each item’s successor.
/// The iterator will yield the `T`s returned from the closure.
///
/// ```
/// use std::iter::successors;
///
/// let powers_of_10 = successors(Some(1_u16), |n| n.checked_mul(10));
/// assert_eq!(powers_of_10.collect::<Vec<_>>(), &[1, 10, 100, 1_000, 10_000]);
/// ```
#[stable(feature = "iter_successors", since = "1.34.0")]
pub fn successors<T, F>(first: Option<T>, succ: F) -> Successors<T, F>
where
    F: FnMut(&T) -> Option<T>,
{
    // If this function returned `impl Iterator<Item=T>`
    // it could be based on `from_fn` and not need a dedicated type.
    // However having a named `Successors<T, F>` type allows it to be `Clone` when `T` and `F` are.
    Successors { next: first, succ }
}

/// A new iterator where each successive item is computed based on the preceding one.
///
/// This `struct` is created by the [`iter::successors()`] function.
/// See its documentation for more.
///
/// [`iter::successors()`]: successors
#[derive(Clone)]
#[stable(feature = "iter_successors", since = "1.34.0")]
pub struct Successors<T, F> {
    next: Option<T>,
    succ: F,
}

#[stable(feature = "iter_successors", since = "1.34.0")]
impl<T, F> Iterator for Successors<T, F>
where
    F: FnMut(&T) -> Option<T>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.next.take()?;
        self.next = (self.succ)(&item);
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.next.is_some() { (1, None) } else { (0, Some(0)) }
    }
}

#[stable(feature = "iter_successors", since = "1.34.0")]
impl<T, F> FusedIterator for Successors<T, F> where F: FnMut(&T) -> Option<T> {}

#[stable(feature = "iter_successors", since = "1.34.0")]
impl<T: fmt::Debug, F> fmt::Debug for Successors<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Successors").field("next", &self.next).finish()
    }
}
