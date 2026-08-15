use crate::fmt;
use crate::iter::FusedIterator;

/// Creates an iterator which, starting from an initial item,
/// computes each successive item from the preceding one.
///
/// This iterator stores an optional item (`Option<T>`) and a successor closure (`impl FnMut(&T) -> Option<T>`).
/// Its `next` method returns the stored optional item and
/// if it is `Some(val)` calls the stored closure on `&val` to compute and store its successor.
/// The iterator will apply the closure successively to the stored option's value until the option is `None`.
/// This also means that once the stored option is `None` it will remain `None`,
/// as the closure will not be called again, so the created iterator is a [`FusedIterator`].
/// The iterator's items will be the initial item and all of its successors as calculated by the successor closure.
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

/// An iterator which, starting from an initial item,
/// computes each successive item from the preceding one.
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
