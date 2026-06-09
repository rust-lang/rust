use super::{IntoIter, VecDeque};

/// Specialization trait used for `VecDeque::from_iter`
pub(super) trait SpecFromIter<T, I> {
    fn spec_from_iter(iter: I) -> Self;
}

impl<T, I> SpecFromIter<T, I> for VecDeque<T>
where
    I: Iterator<Item = T>,
{
    default fn spec_from_iter(iterator: I) -> Self {
        // Since converting is O(1) now, just re-use the `Vec` logic for
        // anything where we can't do something extra-special for `VecDeque`,
        // especially as that could save us some monomorphization work
        // if one uses the same iterators (like slice ones) with both.
        crate::vec::Vec::from_iter(iterator).into()
    }
}

#[cfg(not(test))]
impl<T> SpecFromIter<T, crate::vec::IntoIter<T>> for VecDeque<T> {
    #[inline]
    fn spec_from_iter(iterator: crate::vec::IntoIter<T>) -> Self {
        iterator.into_vecdeque()
    }
}

impl<T> SpecFromIter<T, IntoIter<T>> for VecDeque<T> {
    #[inline]
    fn spec_from_iter(iterator: IntoIter<T>) -> Self {
        iterator.into_vecdeque()
    }
}
