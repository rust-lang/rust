use crate::iter::IntoIterator as RealIntoIterator;
use crate::iter::TrustedRandomAccessNoCoerce;

#[unstable(feature = "trusted_random_access", issue = "none")]
#[doc(hidden)]

pub trait IntoIterator {
    type IntoIter: Iterator;

    #[unstable(feature = "trusted_random_access", issue = "none")]
    // #[cfg_attr(not(bootstrap), lang = "loop_desugar")]
    #[cfg_attr(not(bootstrap), lang = "into_iter")]
    fn into_iter(self) -> Self::IntoIter;
}

impl<C: RealIntoIterator> IntoIterator for C {
    type IntoIter = ForLoopDesugar<C::IntoIter>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ForLoopDesugar { iter: <Self as RealIntoIterator>::into_iter(self), idx: 0 }
    }
}

#[derive(Debug)]
#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
pub struct ForLoopDesugar<I> {
    iter: I,
    idx: usize,
}

#[unstable(feature = "trusted_random_access", issue = "none")]
impl<I: Iterator> Iterator for ForLoopDesugar<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        // self.iter.next_spec(&mut self.idx)
        self.next_spec()
    }
}

trait DesugarSpec<T> {
    fn next_spec(&mut self) -> Option<T>;
}

impl<I, T> DesugarSpec<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn next_spec(&mut self) -> Option<I::Item> {
        self.iter.next()
    }
}

impl<I, T> DesugarSpec<T> for ForLoopDesugar<I>
where
    I: TrustedRandomAccessNoCoerce + Iterator<Item = T>,
{
    #[inline]
    fn next_spec(&mut self) -> Option<I::Item> {
        let idx = self.idx;
        if idx < self.iter.size() {
            // SAFETY: idx can't overflow since size is a usize. idx is always
            // less than size, so the index is always valid.
            unsafe {
                self.idx = idx.unchecked_add(1);
                Some(self.iter.__iterator_get_unchecked(idx))
            }
        } else {
            None
        }
    }
}
