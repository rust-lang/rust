/// Collapses all unit items from an iterator into one.
///
/// This is more useful when combined with higher-level abstractions, like
/// collecting to a `Result<(), E>` where you only care about errors:
///
/// ```
/// use std::io::*;
/// let data = vec![1, 2, 3, 4, 5];
/// let res: Result<()> = data.iter()
///     .map(|x| writeln!(stdout(), "{x}"))
///     .collect();
/// assert!(res.is_ok());
/// ```
#[stable(feature = "unit_from_iter", since = "1.23.0")]
impl FromIterator<()> for () {
    fn from_iter<I: IntoIterator<Item = ()>>(iter: I) -> Self {
        iter.into_iter().for_each(|()| {})
    }
}
