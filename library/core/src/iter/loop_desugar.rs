use crate::iter::IntoIterator;
use crate::iter::TrustedRandomAccessNoCoerce;

#[derive(Debug)]
#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
pub struct ForLoopDesugar<I: Iterator> {
    iter: I,
    idx: usize,
}

impl<I: Iterator> ForLoopDesugar<I> {
    #[inline]
    #[cfg_attr(not(bootstrap), lang = "into_iter")]
    #[cfg_attr(bootstrap, allow(dead_code))]
    pub fn new(it: impl IntoIterator<Item = I::Item, IntoIter = I>) -> Self {
        let mut desugar = ForLoopDesugar { iter: it.into_iter(), idx: 0 };
        desugar.setup();
        desugar
    }

    #[inline]
    #[cfg_attr(not(bootstrap), lang = "next")]
    #[cfg_attr(bootstrap, allow(dead_code))]
    pub fn next(&mut self) -> Option<I::Item> {
        self.next_spec()
    }
}

unsafe impl<#[may_dangle] I: Iterator> Drop for ForLoopDesugar<I> {
    #[inline]
    fn drop(&mut self) {
        self.cleanup();
    }
}

trait DesugarSpec<T> {
    fn setup(&mut self);

    fn next_spec(&mut self) -> Option<T>;

    fn cleanup(&mut self);
}

impl<I, T> DesugarSpec<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn setup(&mut self) {}

    #[inline]
    default fn next_spec(&mut self) -> Option<I::Item> {
        self.iter.next()
    }

    #[inline]
    default fn cleanup(&mut self) {}
}

impl<I, T> DesugarSpec<T> for ForLoopDesugar<I>
where
    I: TrustedRandomAccessNoCoerce + Iterator<Item = T>,
{
    #[inline]
    fn setup(&mut self) {
        let _ = self.iter.advance_by(0);
    }

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

    #[inline]
    fn cleanup(&mut self) {
        if I::NEEDS_CLEANUP {
            self.iter.cleanup(self.idx, true);
        }
    }
}
