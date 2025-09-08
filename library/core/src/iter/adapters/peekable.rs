use crate::iter::adapters::SourceIter;
use crate::iter::{FusedIterator, TrustedLen};
use crate::ops::{ControlFlow, Try};

/// An iterator with a `peek()` that returns an optional reference to the next
/// element.
///
/// This `struct` is created by the [`peekable`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`peekable`]: Iterator::peekable
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "IterPeekable"]
pub struct Peekable<I: Iterator> {
    iter: I,
    /// Remember a peeked value, even if it was None.
    peeked: Option<Option<I::Item>>,
}

impl<I: Iterator> Peekable<I> {
    pub(in crate::iter) fn new(iter: I) -> Peekable<I> {
        Peekable { iter, peeked: None }
    }
}

// Peekable must remember if a None has been seen in the `.peek()` method.
// It ensures that `.peek(); .peek();` or `.peek(); .next();` only advances the
// underlying iterator at most once. This does not by itself make the iterator
// fused.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        match self.peeked.take() {
            Some(v) => v,
            None => self.iter.next(),
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn count(mut self) -> usize {
        match self.peeked.take() {
            Some(None) => 0,
            Some(Some(_)) => 1 + self.iter.count(),
            None => self.iter.count(),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        match self.peeked.take() {
            Some(None) => None,
            Some(v @ Some(_)) if n == 0 => v,
            Some(Some(_)) => self.iter.nth(n - 1),
            None => self.iter.nth(n),
        }
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        let peek_opt = match self.peeked.take() {
            Some(None) => return None,
            Some(v) => v,
            None => None,
        };
        self.iter.last().or(peek_opt)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let peek_len = match self.peeked {
            Some(None) => return (0, Some(0)),
            Some(Some(_)) => 1,
            None => 0,
        };
        let (lo, hi) = self.iter.size_hint();
        let lo = lo.saturating_add(peek_len);
        let hi = match hi {
            Some(x) => x.checked_add(peek_len),
            None => None,
        };
        (lo, hi)
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let acc = match self.peeked.take() {
            Some(None) => return try { init },
            Some(Some(v)) => f(init, v)?,
            None => init,
        };
        self.iter.try_fold(acc, f)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let acc = match self.peeked {
            Some(None) => return init,
            Some(Some(v)) => fold(init, v),
            None => init,
        };
        self.iter.fold(acc, fold)
    }
}

#[stable(feature = "double_ended_peek_iterator", since = "1.38.0")]
impl<I> DoubleEndedIterator for Peekable<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.peeked.as_mut() {
            Some(v @ Some(_)) => self.iter.next_back().or_else(|| v.take()),
            Some(None) => None,
            None => self.iter.next_back(),
        }
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        match self.peeked.take() {
            Some(None) => try { init },
            Some(Some(v)) => match self.iter.try_rfold(init, &mut f).branch() {
                ControlFlow::Continue(acc) => f(acc, v),
                ControlFlow::Break(r) => {
                    self.peeked = Some(Some(v));
                    R::from_residual(r)
                }
            },
            None => self.iter.try_rfold(init, f),
        }
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        match self.peeked {
            Some(None) => init,
            Some(Some(v)) => {
                let acc = self.iter.rfold(init, &mut fold);
                fold(acc, v)
            }
            None => self.iter.rfold(init, fold),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator> ExactSizeIterator for Peekable<I> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator> FusedIterator for Peekable<I> {}

impl<I: Iterator> Peekable<I> {
    /// Returns a reference to the next() value without advancing the iterator.
    ///
    /// Like [`next`], if there is a value, it is wrapped in a `Some(T)`.
    /// But if the iteration is over, `None` is returned.
    ///
    /// [`next`]: Iterator::next
    ///
    /// Because `peek()` returns a reference, and many iterators iterate over
    /// references, there can be a possibly confusing situation where the
    /// return value is a double reference. You can see this effect in the
    /// examples below.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let xs = [1, 2, 3];
    ///
    /// let mut iter = xs.iter().peekable();
    ///
    /// // peek() lets us see into the future
    /// assert_eq!(iter.peek(), Some(&&1));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// assert_eq!(iter.next(), Some(&2));
    ///
    /// // The iterator does not advance even if we `peek` multiple times
    /// assert_eq!(iter.peek(), Some(&&3));
    /// assert_eq!(iter.peek(), Some(&&3));
    ///
    /// assert_eq!(iter.next(), Some(&3));
    ///
    /// // After the iterator is finished, so is `peek()`
    /// assert_eq!(iter.peek(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn peek(&mut self) -> Option<&I::Item> {
        let iter = &mut self.iter;
        self.peeked.get_or_insert_with(|| iter.next()).as_ref()
    }

    /// Returns a mutable reference to the next() value without advancing the iterator.
    ///
    /// Like [`next`], if there is a value, it is wrapped in a `Some(T)`.
    /// But if the iteration is over, `None` is returned.
    ///
    /// Because `peek_mut()` returns a reference, and many iterators iterate over
    /// references, there can be a possibly confusing situation where the
    /// return value is a double reference. You can see this effect in the examples
    /// below.
    ///
    /// [`next`]: Iterator::next
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut iter = [1, 2, 3].iter().peekable();
    ///
    /// // Like with `peek()`, we can see into the future without advancing the iterator.
    /// assert_eq!(iter.peek_mut(), Some(&mut &1));
    /// assert_eq!(iter.peek_mut(), Some(&mut &1));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// // Peek into the iterator and set the value behind the mutable reference.
    /// if let Some(p) = iter.peek_mut() {
    ///     assert_eq!(*p, &2);
    ///     *p = &5;
    /// }
    ///
    /// // The value we put in reappears as the iterator continues.
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![&5, &3]);
    /// ```
    #[inline]
    #[stable(feature = "peekable_peek_mut", since = "1.53.0")]
    pub fn peek_mut(&mut self) -> Option<&mut I::Item> {
        let iter = &mut self.iter;
        self.peeked.get_or_insert_with(|| iter.next()).as_mut()
    }

    /// Consume and return the next value of this iterator if a condition is true.
    ///
    /// If `func` returns `true` for the next value of this iterator, consume and return it.
    /// Otherwise, return `None`.
    ///
    /// # Examples
    /// Consume a number if it's equal to 0.
    /// ```
    /// let mut iter = (0..5).peekable();
    /// // The first item of the iterator is 0; consume it.
    /// assert_eq!(iter.next_if(|&x| x == 0), Some(0));
    /// // The next item returned is now 1, so `next_if` will return `None`.
    /// assert_eq!(iter.next_if(|&x| x == 0), None);
    /// // `next_if` retains the next item if the predicate evaluates to `false` for it.
    /// assert_eq!(iter.next(), Some(1));
    /// ```
    ///
    /// Consume any number less than 10.
    /// ```
    /// let mut iter = (1..20).peekable();
    /// // Consume all numbers less than 10
    /// while iter.next_if(|&x| x < 10).is_some() {}
    /// // The next value returned will be 10
    /// assert_eq!(iter.next(), Some(10));
    /// ```
    #[stable(feature = "peekable_next_if", since = "1.51.0")]
    pub fn next_if(&mut self, func: impl FnOnce(&I::Item) -> bool) -> Option<I::Item> {
        match self.next() {
            Some(matched) if func(&matched) => Some(matched),
            other => {
                // Since we called `self.next()`, we consumed `self.peeked`.
                assert!(self.peeked.is_none());
                self.peeked = Some(other);
                None
            }
        }
    }

    /// Consume and return the next item if it is equal to `expected`.
    ///
    /// # Example
    /// Consume a number if it's equal to 0.
    /// ```
    /// let mut iter = (0..5).peekable();
    /// // The first item of the iterator is 0; consume it.
    /// assert_eq!(iter.next_if_eq(&0), Some(0));
    /// // The next item returned is now 1, so `next_if_eq` will return `None`.
    /// assert_eq!(iter.next_if_eq(&0), None);
    /// // `next_if_eq` retains the next item if it was not equal to `expected`.
    /// assert_eq!(iter.next(), Some(1));
    /// ```
    #[stable(feature = "peekable_next_if", since = "1.51.0")]
    pub fn next_if_eq<T>(&mut self, expected: &T) -> Option<I::Item>
    where
        T: ?Sized,
        I::Item: PartialEq<T>,
    {
        self.next_if(|next| next == expected)
    }

    /// Consumes the next value of this iterator and applies a function `f` on it,
    /// returning the result if the closure returns `Ok`.
    ///
    /// Otherwise if the closure returns `Err` the value is put back for the next iteration.
    ///
    /// The content of the `Err` variant is typically the original value of the closure,
    /// but this is not required. If a different value is returned,
    /// the next `peek()` or `next()` call will result in this new value.
    /// This is similar to modifying the output of `peek_mut()`.
    ///
    /// If the closure panics, the next value will always be consumed and dropped
    /// even if the panic is caught, because the closure never returned an `Err` value to put back.
    ///
    /// # Examples
    ///
    /// Parse the leading decimal number from an iterator of characters.
    /// ```
    /// #![feature(peekable_next_if_map)]
    /// let mut iter = "125 GOTO 10".chars().peekable();
    /// let mut line_num = 0_u32;
    /// while let Some(digit) = iter.next_if_map(|c| c.to_digit(10).ok_or(c)) {
    ///     line_num = line_num * 10 + digit;
    /// }
    /// assert_eq!(line_num, 125);
    /// assert_eq!(iter.collect::<String>(), " GOTO 10");
    /// ```
    ///
    /// Matching custom types.
    /// ```
    /// #![feature(peekable_next_if_map)]
    ///
    /// #[derive(Debug, PartialEq, Eq)]
    /// enum Node {
    ///     Comment(String),
    ///     Red(String),
    ///     Green(String),
    ///     Blue(String),
    /// }
    ///
    /// /// Combines all consecutive `Comment` nodes into a single one.
    /// fn combine_comments(nodes: Vec<Node>) -> Vec<Node> {
    ///     let mut result = Vec::with_capacity(nodes.len());
    ///     let mut iter = nodes.into_iter().peekable();
    ///     let mut comment_text = None::<String>;
    ///     loop {
    ///         // Typically the closure in .next_if_map() matches on the input,
    ///         //  extracts the desired pattern into an `Ok`,
    ///         //  and puts the rest into an `Err`.
    ///         while let Some(text) = iter.next_if_map(|node| match node {
    ///             Node::Comment(text) => Ok(text),
    ///             other => Err(other),
    ///         }) {
    ///             comment_text.get_or_insert_default().push_str(&text);
    ///         }
    ///
    ///         if let Some(text) = comment_text.take() {
    ///             result.push(Node::Comment(text));
    ///         }
    ///         if let Some(node) = iter.next() {
    ///             result.push(node);
    ///         } else {
    ///             break;
    ///         }
    ///     }
    ///     result
    /// }
    ///# assert_eq!( // hiding the test to avoid cluttering the documentation.
    ///#     combine_comments(vec![
    ///#         Node::Comment("The".to_owned()),
    ///#         Node::Comment("Quick".to_owned()),
    ///#         Node::Comment("Brown".to_owned()),
    ///#         Node::Red("Fox".to_owned()),
    ///#         Node::Green("Jumped".to_owned()),
    ///#         Node::Comment("Over".to_owned()),
    ///#         Node::Blue("The".to_owned()),
    ///#         Node::Comment("Lazy".to_owned()),
    ///#         Node::Comment("Dog".to_owned()),
    ///#     ]),
    ///#     vec![
    ///#         Node::Comment("TheQuickBrown".to_owned()),
    ///#         Node::Red("Fox".to_owned()),
    ///#         Node::Green("Jumped".to_owned()),
    ///#         Node::Comment("Over".to_owned()),
    ///#         Node::Blue("The".to_owned()),
    ///#         Node::Comment("LazyDog".to_owned()),
    ///#     ],
    ///# )
    /// ```
    #[unstable(feature = "peekable_next_if_map", issue = "143702")]
    pub fn next_if_map<R>(&mut self, f: impl FnOnce(I::Item) -> Result<R, I::Item>) -> Option<R> {
        let unpeek = if let Some(item) = self.next() {
            match f(item) {
                Ok(result) => return Some(result),
                Err(item) => Some(item),
            }
        } else {
            None
        };
        self.peeked = Some(unpeek);
        None
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Peekable<I> where I: TrustedLen {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: Iterator> SourceIter for Peekable<I>
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
