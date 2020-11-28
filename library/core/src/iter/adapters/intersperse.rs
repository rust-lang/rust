use super::Peekable;

/// An iterator adapter that places a separator between all elements.
#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
#[derive(Debug, Clone)]
pub struct Intersperse<I: Iterator>
where
    I::Item: Clone,
{
    separator: I::Item,
    iter: Peekable<I>,
    needs_sep: bool,
}

impl<I: Iterator> Intersperse<I>
where
    I::Item: Clone,
{
    pub(in crate::iter) fn new(iter: I, separator: I::Item) -> Self {
        Self { iter: iter.peekable(), separator, needs_sep: false }
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I> Iterator for Intersperse<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.needs_sep && self.iter.peek().is_some() {
            self.needs_sep = false;
            Some(self.separator.clone())
        } else {
            self.needs_sep = true;
            self.iter.next()
        }
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;

        // Use `peek()` first to avoid calling `next()` on an empty iterator.
        if !self.needs_sep || self.iter.peek().is_some() {
            if let Some(x) = self.iter.next() {
                accum = f(accum, x);
            }
        }

        let element = &self.separator;

        self.iter.fold(accum, |mut accum, x| {
            accum = f(accum, element.clone());
            accum = f(accum, x);
            accum
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        let next_is_elem = !self.needs_sep;
        let lo = lo.saturating_sub(next_is_elem as usize).saturating_add(lo);
        let hi = match hi {
            Some(hi) => hi.saturating_sub(next_is_elem as usize).checked_add(hi),
            None => None,
        };
        (lo, hi)
    }
}
