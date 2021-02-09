use crate::fmt;

/// An iterator over the mapped windows of another iterator.
///
/// This `struct` is created by the [`Iterator::map_windows`]. See its
/// documentation for more information.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "none")]
pub struct MapWindows<I: Iterator, F, const N: usize> {
    iter: I,
    f: F,
    buffer: Option<[I::Item; N]>,
}

impl<I: Iterator, F, const N: usize> MapWindows<I, F, N> {
    pub(in crate::iter) fn new(mut iter: I, f: F) -> Self {
        assert!(N > 0, "array in `Iterator::map_windows` must contain more than 0 elements");

        let buffer = crate::array::collect_into_array(&mut iter);
        Self { iter, f, buffer }
    }
}

#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "none")]
impl<I, F, R, const N: usize> Iterator for MapWindows<I, F, N>
where
    I: Iterator,
    F: FnMut(&[I::Item; N]) -> R,
{
    type Item = R;
    fn next(&mut self) -> Option<Self::Item> {
        let buffer = self.buffer.as_mut()?;
        let out = (self.f)(buffer);

        // Advance iterator
        if let Some(next) = self.iter.next() {
            buffer.rotate_left(1);
            buffer[N - 1] = next;
        } else {
            self.buffer = None;
        }

        Some(out)
    }
}

#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "none")]
impl<I: Iterator + fmt::Debug, F, const N: usize> fmt::Debug for MapWindows<I, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapWindows").field("iter", &self.iter).finish()
    }
}
