use crate::iter::Fuse;

/// An iterator adapter that places a separator between all elements.
#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "none")]
#[derive(Debug, Clone)]
pub struct Intersperse<I: Iterator>
where
    I::Item: Clone,
{
    element: I::Item,
    iter: Fuse<I>,
    peek: Option<I::Item>,
}

impl<I: Iterator> Intersperse<I>
where
    I::Item: Clone,
{
    pub(in super::super) fn new(iter: I, separator: I::Item) -> Self {
        let mut iter = iter.fuse();
        Self { peek: iter.next(), iter, element: separator }
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "none")]
impl<I> Iterator for Intersperse<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if let Some(item) = self.peek.take() {
            Some(item)
        } else {
            self.peek = Some(self.iter.next()?);
            Some(self.element.clone())
        }
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;

        if let Some(x) = self.peek.take() {
            accum = f(accum, x);
        }

        let element = &self.element;

        self.iter.fold(accum, |accum, x| {
            let accum = f(accum, element.clone());
            let accum = f(accum, x);
            accum
        })
    }
}
