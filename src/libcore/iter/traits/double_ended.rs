use crate::ops::Try;
use crate::iter::LoopState;

/// An iterator able to yield elements from both ends.
///
/// Something that implements `DoubleEndedIterator` has one extra capability
/// over something that implements [`Iterator`]: the ability to also take
/// `Item`s from the back, as well as the front.
///
/// It is important to note that both back and forth work on the same range,
/// and do not cross: iteration is over when they meet in the middle.
///
/// In a similar fashion to the [`Iterator`] protocol, once a
/// `DoubleEndedIterator` returns `None` from a `next_back()`, calling it again
/// may or may not ever return `Some` again. `next()` and `next_back()` are
/// interchangeable for this purpose.
///
/// [`Iterator`]: trait.Iterator.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let numbers = vec![1, 2, 3, 4, 5, 6];
///
/// let mut iter = numbers.iter();
///
/// assert_eq!(Some(&1), iter.next());
/// assert_eq!(Some(&6), iter.next_back());
/// assert_eq!(Some(&5), iter.next_back());
/// assert_eq!(Some(&2), iter.next());
/// assert_eq!(Some(&3), iter.next());
/// assert_eq!(Some(&4), iter.next());
/// assert_eq!(None, iter.next());
/// assert_eq!(None, iter.next_back());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait DoubleEndedIterator: Iterator {
    /// Removes and returns an element from the end of the iterator.
    ///
    /// Returns `None` when there are no more elements.
    ///
    /// The [trait-level] docs contain more details.
    ///
    /// [trait-level]: trait.DoubleEndedIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let numbers = vec![1, 2, 3, 4, 5, 6];
    ///
    /// let mut iter = numbers.iter();
    ///
    /// assert_eq!(Some(&1), iter.next());
    /// assert_eq!(Some(&6), iter.next_back());
    /// assert_eq!(Some(&5), iter.next_back());
    /// assert_eq!(Some(&2), iter.next());
    /// assert_eq!(Some(&3), iter.next());
    /// assert_eq!(Some(&4), iter.next());
    /// assert_eq!(None, iter.next());
    /// assert_eq!(None, iter.next_back());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next_back(&mut self) -> Option<Self::Item>;

    /// Returns the `n`th element from the end of the iterator.
    ///
    /// This is essentially the reversed version of [`nth`]. Although like most indexing
    /// operations, the count starts from zero, so `nth_back(0)` returns the first value fro
    /// the end, `nth_back(1)` the second, and so on.
    ///
    /// Note that all elements between the end and the returned element will be
    /// consumed, including the returned element. This also means that calling
    /// `nth_back(0)` multiple times on the same iterator will return different
    /// elements.
    ///
    /// `nth_back()` will return [`None`] if `n` is greater than or equal to the length of the
    /// iterator.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`nth`]: ../../std/iter/trait.Iterator.html#method.nth
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().nth_back(2), Some(&1));
    /// ```
    ///
    /// Calling `nth_back()` multiple times doesn't rewind the iterator:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.nth_back(1), Some(&2));
    /// assert_eq!(iter.nth_back(1), None);
    /// ```
    ///
    /// Returning `None` if there are less than `n + 1` elements:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().nth_back(10), None);
    /// ```
    #[inline]
    #[stable(feature = "iter_nth_back", since = "1.37.0")]
    fn nth_back(&mut self, mut n: usize) -> Option<Self::Item> {
        for x in self.rev() {
            if n == 0 { return Some(x) }
            n -= 1;
        }
        None
    }

    /// This is the reverse version of [`try_fold()`]: it takes elements
    /// starting from the back of the iterator.
    ///
    /// [`try_fold()`]: trait.Iterator.html#method.try_fold
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = ["1", "2", "3"];
    /// let sum = a.iter()
    ///     .map(|&s| s.parse::<i32>())
    ///     .try_rfold(0, |acc, x| x.and_then(|y| Ok(acc + y)));
    /// assert_eq!(sum, Ok(6));
    /// ```
    ///
    /// Short-circuiting:
    ///
    /// ```
    /// let a = ["1", "rust", "3"];
    /// let mut it = a.iter();
    /// let sum = it
    ///     .by_ref()
    ///     .map(|&s| s.parse::<i32>())
    ///     .try_rfold(0, |acc, x| x.and_then(|y| Ok(acc + y)));
    /// assert!(sum.is_err());
    ///
    /// // Because it short-circuited, the remaining elements are still
    /// // available through the iterator.
    /// assert_eq!(it.next_back(), Some(&"1"));
    /// ```
    #[inline]
    #[stable(feature = "iterator_try_fold", since = "1.27.0")]
    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok=B>
    {
        let mut accum = init;
        while let Some(x) = self.next_back() {
            accum = f(accum, x)?;
        }
        Try::from_ok(accum)
    }

    /// An iterator method that reduces the iterator's elements to a single,
    /// final value, starting from the back.
    ///
    /// This is the reverse version of [`fold()`]: it takes elements starting from
    /// the back of the iterator.
    ///
    /// `rfold()` takes two arguments: an initial value, and a closure with two
    /// arguments: an 'accumulator', and an element. The closure returns the value that
    /// the accumulator should have for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first
    /// call.
    ///
    /// After applying this closure to every element of the iterator, `rfold()`
    /// returns the accumulator.
    ///
    /// This operation is sometimes called 'reduce' or 'inject'.
    ///
    /// Folding is useful whenever you have a collection of something, and want
    /// to produce a single value from it.
    ///
    /// [`fold()`]: trait.Iterator.html#method.fold
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// // the sum of all of the elements of a
    /// let sum = a.iter()
    ///            .rfold(0, |acc, &x| acc + x);
    ///
    /// assert_eq!(sum, 6);
    /// ```
    ///
    /// This example builds a string, starting with an initial value
    /// and continuing with each element from the back until the front:
    ///
    /// ```
    /// let numbers = [1, 2, 3, 4, 5];
    ///
    /// let zero = "0".to_string();
    ///
    /// let result = numbers.iter().rfold(zero, |acc, &x| {
    ///     format!("({} + {})", x, acc)
    /// });
    ///
    /// assert_eq!(result, "(1 + (2 + (3 + (4 + (5 + 0)))))");
    /// ```
    #[inline]
    #[stable(feature = "iter_rfold", since = "1.27.0")]
    fn rfold<B, F>(mut self, accum: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.try_rfold(accum, move |acc, x| Ok::<B, !>(f(acc, x))).unwrap()
    }

    /// Searches for an element of an iterator from the back that satisfies a predicate.
    ///
    /// `rfind()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, starting at the end, and if any
    /// of them return `true`, then `rfind()` returns [`Some(element)`]. If they all return
    /// `false`, it returns [`None`].
    ///
    /// `rfind()` is short-circuiting; in other words, it will stop processing
    /// as soon as the closure returns `true`.
    ///
    /// Because `rfind()` takes a reference, and many iterators iterate over
    /// references, this leads to a possibly confusing situation where the
    /// argument is a double reference. You can see this effect in the
    /// examples below, with `&&x`.
    ///
    /// [`Some(element)`]: ../../std/option/enum.Option.html#variant.Some
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.iter().rfind(|&&x| x == 2), Some(&2));
    ///
    /// assert_eq!(a.iter().rfind(|&&x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.rfind(|&&x| x == 2), Some(&2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next_back(), Some(&1));
    /// ```
    #[inline]
    #[stable(feature = "iter_rfind", since = "1.27.0")]
    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool
    {
        self.try_rfold((), move |(), x| {
            if predicate(&x) { LoopState::Break(x) }
            else { LoopState::Continue(()) }
        }).break_value()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for &'a mut I {
    fn next_back(&mut self) -> Option<I::Item> {
        (**self).next_back()
    }
    fn nth_back(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth_back(n)
    }
}
