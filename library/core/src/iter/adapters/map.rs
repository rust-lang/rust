use crate::const_closure::ConstFnMutClosure;
use crate::fmt;
use crate::iter::adapters::{
    zip::try_get_unchecked, SourceIter, TrustedRandomAccess, TrustedRandomAccessNoCoerce,
};
use crate::iter::{FusedIterator, InPlaceIterable, TrustedLen};
use crate::marker::Destruct;
use crate::ops::Try;

/// An iterator that maps the values of `iter` with `f`.
///
/// This `struct` is created by the [`map`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`map`]: Iterator::map
/// [`Iterator`]: trait.Iterator.html
///
/// # Notes about side effects
///
/// The [`map`] iterator implements [`DoubleEndedIterator`], meaning that
/// you can also [`map`] backwards:
///
/// ```rust
/// let v: Vec<i32> = [1, 2, 3].into_iter().map(|x| x + 1).rev().collect();
///
/// assert_eq!(v, [4, 3, 2]);
/// ```
///
/// [`DoubleEndedIterator`]: trait.DoubleEndedIterator.html
///
/// But if your closure has state, iterating backwards may act in a way you do
/// not expect. Let's go through an example. First, in the forward direction:
///
/// ```rust
/// let mut c = 0;
///
/// for pair in ['a', 'b', 'c'].into_iter()
///                                .map(|letter| { c += 1; (letter, c) }) {
///     println!("{pair:?}");
/// }
/// ```
///
/// This will print `('a', 1), ('b', 2), ('c', 3)`.
///
/// Now consider this twist where we add a call to `rev`. This version will
/// print `('c', 1), ('b', 2), ('a', 3)`. Note that the letters are reversed,
/// but the values of the counter still go in order. This is because `map()` is
/// still being called lazily on each item, but we are popping items off the
/// back of the vector now, instead of shifting them from the front.
///
/// ```rust
/// let mut c = 0;
///
/// for pair in ['a', 'b', 'c'].into_iter()
///                                .map(|letter| { c += 1; (letter, c) })
///                                .rev() {
///     println!("{pair:?}");
/// }
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Map<I, F> {
    // Used for `SplitWhitespace` and `SplitAsciiWhitespace` `as_str` methods
    pub(crate) iter: I,
    f: F,
}

impl<I, F> Map<I, F> {
    pub(in crate::iter) const fn new(iter: I, f: F) -> Map<I, F> {
        Map { iter, f }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, F> fmt::Debug for Map<I, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Map").field("iter", &self.iter).finish()
    }
}

const fn map_fold<F, G, T, B, Acc>(x: &mut (F, G), (acc, elt): (Acc, T)) -> Acc
where
    F: ~const FnMut(T) -> B,
    G: ~const FnMut(Acc, B) -> Acc,
{
    x.1(acc, x.0(elt))
}

const fn map_try_fold<F, G, T, B, Acc, R>(x: &mut (F, G), (acc, elt): (Acc, T)) -> R
where
    F: ~const FnMut(T) -> B,
    G: ~const FnMut(Acc, B) -> R,
{
    x.1(acc, x.0(elt))
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_iter", issue = "92476")]
impl<B, I: ~const Iterator, F> const Iterator for Map<I, F>
where
    F: ~const FnMut(I::Item) -> B,
    F: ~const Destruct,
    I: ~const Destruct,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().map(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn try_fold<Acc, G, R>(&mut self, init: Acc, g: G) -> R
    where
        Self: Sized,
        G: ~const FnMut(Acc, Self::Item) -> R + ~const Destruct,
        R: ~const Try<Output = Acc>,
    {
        let mut tup = (&mut self.f, g);
        self.iter.try_fold(init, ConstFnMutClosure::new(&mut tup, map_try_fold))
    }

    fn fold<Acc, G>(mut self, init: Acc, g: G) -> Acc
    where
        G: ~const FnMut(Acc, Self::Item) -> Acc + ~const Destruct,
        Self: ~const Destruct,
    {
        let mut tup = (&mut self.f, g);
        self.iter.fold(init, ConstFnMutClosure::new(&mut tup, map_fold))
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> B
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { (self.f)(try_get_unchecked(&mut self.iter, idx)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: DoubleEndedIterator, F> DoubleEndedIterator for Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        self.iter.next_back().map(&mut self.f)
    }

    fn try_rfold<Acc, G, R>(&mut self, init: Acc, g: G) -> R
    where
        Self: Sized,
        G: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        let mut tup = (&mut self.f, g);
        self.iter.try_rfold(init, ConstFnMutClosure::new(&mut tup, map_try_fold))
    }

    fn rfold<Acc, G>(mut self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut tup = (&mut self.f, g);
        self.iter.rfold(init, ConstFnMutClosure::new(&mut tup, map_fold))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: ExactSizeIterator, F> ExactSizeIterator for Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<B, I: FusedIterator, F> FusedIterator for Map<I, F> where F: FnMut(I::Item) -> B {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<B, I, F> TrustedLen for Map<I, F>
where
    I: TrustedLen,
    F: FnMut(I::Item) -> B,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I, F> TrustedRandomAccess for Map<I, F> where I: TrustedRandomAccess {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I, F> TrustedRandomAccessNoCoerce for Map<I, F>
where
    I: TrustedRandomAccessNoCoerce,
{
    const MAY_HAVE_SIDE_EFFECT: bool = true;
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, F> SourceIter for Map<I, F>
where
    I: SourceIter,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<B, I: InPlaceIterable, F> InPlaceIterable for Map<I, F> where F: FnMut(I::Item) -> B {}
