use crate::iter::IntoIterator;
use crate::iter::{
    TrustedRandomAccess, TrustedRandomAccessNeedsCleanup, TrustedRandomAccessNeedsForwardSetup,
};

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

trait DesugarNext<T> {
    fn next_spec(&mut self) -> Option<T>;
}

trait DesugarSetup<T> {
    fn setup(&mut self);
}

trait DesugarCleanup<T> {
    fn cleanup(&mut self);
}

impl<I, T> DesugarNext<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn next_spec(&mut self) -> Option<I::Item> {
        self.iter.next()
    }
}

impl<I, T> DesugarNext<T> for ForLoopDesugar<I>
where
    I: TrustedRandomAccess + Iterator<Item = T>,
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

impl<I, T> DesugarSetup<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn setup(&mut self) {}
}

impl<I, T> DesugarSetup<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T> + TrustedRandomAccess + TrustedRandomAccessNeedsForwardSetup,
{
    #[inline]
    fn setup(&mut self) {
        let _ = self.iter.advance_by(0);
    }
}

impl<I, T> DesugarCleanup<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T>,
{
    #[inline]
    default fn cleanup(&mut self) {}
}

impl<I, T> DesugarCleanup<T> for ForLoopDesugar<I>
where
    I: Iterator<Item = T> + TrustedRandomAccessNeedsCleanup + TrustedRandomAccess,
{
    #[inline]
    fn cleanup(&mut self) {
        self.iter.cleanup(self.idx, true);
    }
}
