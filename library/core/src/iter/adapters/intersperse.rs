use super::Peekable;

/// An iterator adapter that places a separator between all elements.
///
/// This `struct` is created by [`Iterator::intersperse`]. See its documentation
/// for more information.
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

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let separator = self.separator;
        intersperse_fold(self.iter, init, f, move || separator.clone(), self.needs_sep)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        intersperse_size_hint(&self.iter, self.needs_sep)
    }
}

/// An iterator adapter that places a separator between all elements.
///
/// This `struct` is created by [`Iterator::intersperse_with`]. See its
/// documentation for more information.
#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
pub struct IntersperseWith<I, G>
where
    I: Iterator,
{
    separator: G,
    iter: Peekable<I>,
    needs_sep: bool,
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> crate::fmt::Debug for IntersperseWith<I, G>
where
    I: Iterator + crate::fmt::Debug,
    I::Item: crate::fmt::Debug,
    G: crate::fmt::Debug,
{
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        f.debug_struct("IntersperseWith")
            .field("separator", &self.separator)
            .field("iter", &self.iter)
            .field("needs_sep", &self.needs_sep)
            .finish()
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> crate::clone::Clone for IntersperseWith<I, G>
where
    I: Iterator + crate::clone::Clone,
    I::Item: crate::clone::Clone,
    G: Clone,
{
    fn clone(&self) -> Self {
        IntersperseWith {
            separator: self.separator.clone(),
            iter: self.iter.clone(),
            needs_sep: self.needs_sep.clone(),
        }
    }
}

impl<I, G> IntersperseWith<I, G>
where
    I: Iterator,
    G: FnMut() -> I::Item,
{
    pub(in crate::iter) fn new(iter: I, separator: G) -> Self {
        Self { iter: iter.peekable(), separator, needs_sep: false }
    }
}

#[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
impl<I, G> Iterator for IntersperseWith<I, G>
where
    I: Iterator,
    G: FnMut() -> I::Item,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.needs_sep && self.iter.peek().is_some() {
            self.needs_sep = false;
            Some((self.separator)())
        } else {
            self.needs_sep = true;
            self.iter.next()
        }
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        intersperse_fold(self.iter, init, f, self.separator, self.needs_sep)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        intersperse_size_hint(&self.iter, self.needs_sep)
    }
}

fn intersperse_size_hint<I>(iter: &I, needs_sep: bool) -> (usize, Option<usize>)
where
    I: Iterator,
{
    let (lo, hi) = iter.size_hint();
    let next_is_elem = !needs_sep;
    (
        lo.saturating_sub(next_is_elem as usize).saturating_add(lo),
        hi.and_then(|hi| hi.saturating_sub(next_is_elem as usize).checked_add(hi)),
    )
}

fn intersperse_fold<I, B, F, G>(
    mut iter: I,
    init: B,
    mut f: F,
    mut separator: G,
    needs_sep: bool,
) -> B
where
    I: Iterator,
    F: FnMut(B, I::Item) -> B,
    G: FnMut() -> I::Item,
{
    let mut accum = init;

    if !needs_sep {
        if let Some(x) = iter.next() {
            accum = f(accum, x);
        } else {
            return accum;
        }
    }

    iter.fold(accum, |mut accum, x| {
        accum = f(accum, separator());
        accum = f(accum, x);
        accum
    })
}
