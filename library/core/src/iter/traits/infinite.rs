/// An iterator that will always produce more items.
///
/// This is useful in order to have more types implement `ExactSizeIterator`.
///
/// ```
/// [1, 2, 3].zip(repeat(4))
/// ```
///
/// This is an iterator with a known exact size of three.
///
/// In order to propagate this information to the compiler, we
/// have:
///
/// ```text
/// impl<A: Clone> InfiniteIterator for Repeat<A> {}
/// impl<A: Clone, B: ExactSizeIterator> ExactSizeIterator for Zip<Repeat<A>, B> {}
/// impl<A: ExactSizeIterator, B: Clone> ExactSizeIterator for Zip<A, Repeat<B>> {}
/// ```
#[allow(multiple_supertrait_upcastable)]
#[stable(feature = "infinite_iterator_trait", since = "CURRENT_RUSTC_VERSION")]
pub trait InfiniteIterator: Iterator + !ExactSizeIterator {}

#[stable(feature = "infinite_iterator_trait", since = "CURRENT_RUSTC_VERSION")]
impl<'a, I: InfiniteIterator + ?Sized> InfiniteIterator for &'a mut I where
    &'a mut I: !ExactSizeIterator
{
}
