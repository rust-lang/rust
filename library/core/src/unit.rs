use crate::iter::FromIterator;

/// Drains all items from an iterator.
///
/// This is useful to run an iterator to completion when you don't
/// care about the result, or to collect into a `Result<(), E>` when
/// you only care about errors:
///
/// ```
/// use std::io::*;
/// let data = vec![1, 2, 3, 4, 5];
/// let res: Result<()> = data.iter()
///     .map(|x| writeln!(stdout(), "{}", x))
///     .collect();
/// assert!(res.is_ok());
/// ```
#[stable(feature = "unit_from_iter", since = "1.23.0")]
impl<T> FromIterator<T> for () {
    fn from_iter<A: IntoIterator<Item = T>>(iter: A) -> Self {
        iter.into_iter().for_each(|_| {})
    }
}
