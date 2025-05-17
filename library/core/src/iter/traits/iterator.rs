use super::super::{
    ArrayChunks, ByRefSized, Chain, Cloned, Copied, Cycle, Enumerate, Filter, FilterMap, FlatMap,
    Flatten, Fuse, Inspect, Intersperse, IntersperseWith, Map, MapWhile, MapWindows, Peekable,
    Product, Rev, Scan, Skip, SkipWhile, StepBy, Sum, Take, TakeWhile, TrustedRandomAccessNoCoerce,
    Zip, try_process,
};
use crate::array;
use crate::cmp::{self, Ordering};
use crate::num::NonZero;
use crate::ops::{ChangeOutputType, ControlFlow, FromResidual, Residual, Try};

fn _assert_is_dyn_compatible(_: &dyn Iterator<Item = ()>) {}

/// A trait for dealing with iterators.
///
/// This is the main iterator trait. For more about the concept of iterators
/// generally, please see the [module-level documentation]. In particular, you
/// may want to know how to [implement `Iterator`][impl].
///
/// [module-level documentation]: crate::iter
/// [impl]: crate::iter#implementing-iterator
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    on(
        Self = "core::ops::range::RangeTo<Idx>",
        note = "you might have meant to use a bounded `Range`"
    ),
    on(
        Self = "core::ops::range::RangeToInclusive<Idx>",
        note = "you might have meant to use a bounded `RangeInclusive`"
    ),
    label = "`{Self}` is not an iterator",
    message = "`{Self}` is not an iterator"
)]
#[doc(notable_trait)]
#[lang = "iterator"]
#[rustc_diagnostic_item = "Iterator"]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub trait Iterator {
    /// The type of the elements being iterated over.
    #[rustc_diagnostic_item = "IteratorItem"]
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Advances the iterator and returns the next value.
    ///
    /// Returns [`None`] when iteration is finished. Individual iterator
    /// implementations may choose to resume iteration, and so calling `next()`
    /// again may or may not eventually start returning [`Some(Item)`] again at some
    /// point.
    ///
    /// [`Some(Item)`]: Some
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// // A call to next() returns the next value...
    /// assert_eq!(Some(1), iter.next());
    /// assert_eq!(Some(2), iter.next());
    /// assert_eq!(Some(3), iter.next());
    ///
    /// // ... and then None once it's over.
    /// assert_eq!(None, iter.next());
    ///
    /// // More calls may or may not return `None`. Here, they always will.
    /// assert_eq!(None, iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[lang = "next"]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next(&mut self) -> Option<Self::Item>;

    /// Advances the iterator and returns an array containing the next `N` values.
    ///
    /// If there are not enough elements to fill the array then `Err` is returned
    /// containing an iterator over the remaining elements.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_next_chunk)]
    ///
    /// let mut iter = "lorem".chars();
    ///
    /// assert_eq!(iter.next_chunk().unwrap(), ['l', 'o']);              // N is inferred as 2
    /// assert_eq!(iter.next_chunk().unwrap(), ['r', 'e', 'm']);         // N is inferred as 3
    /// assert_eq!(iter.next_chunk::<4>().unwrap_err().as_slice(), &[]); // N is explicitly 4
    /// ```
    ///
    /// Split a string and get the first three items.
    ///
    /// ```
    /// #![feature(iter_next_chunk)]
    ///
    /// let quote = "not all those who wander are lost";
    /// let [first, second, third] = quote.split_whitespace().next_chunk().unwrap();
    /// assert_eq!(first, "not");
    /// assert_eq!(second, "all");
    /// assert_eq!(third, "those");
    /// ```
    #[inline]
    #[unstable(feature = "iter_next_chunk", reason = "recently added", issue = "98326")]
    fn next_chunk<const N: usize>(
        &mut self,
    ) -> Result<[Self::Item; N], array::IntoIter<Self::Item, N>>
    where
        Self: Sized,
    {
        array::iter_next_chunk(self)
    }

    /// Returns the bounds on the remaining length of the iterator.
    ///
    /// Specifically, `size_hint()` returns a tuple where the first element
    /// is the lower bound, and the second element is the upper bound.
    ///
    /// The second half of the tuple that is returned is an <code>[Option]<[usize]></code>.
    /// A [`None`] here means that either there is no known upper bound, or the
    /// upper bound is larger than [`usize`].
    ///
    /// # Implementation notes
    ///
    /// It is not enforced that an iterator implementation yields the declared
    /// number of elements. A buggy iterator may yield less than the lower bound
    /// or more than the upper bound of elements.
    ///
    /// `size_hint()` is primarily intended to be used for optimizations such as
    /// reserving space for the elements of the iterator, but must not be
    /// trusted to e.g., omit bounds checks in unsafe code. An incorrect
    /// implementation of `size_hint()` should not lead to memory safety
    /// violations.
    ///
    /// That said, the implementation should provide a correct estimation,
    /// because otherwise it would be a violation of the trait's protocol.
    ///
    /// The default implementation returns <code>(0, [None])</code> which is correct for any
    /// iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let mut iter = a.iter();
    ///
    /// assert_eq!((3, Some(3)), iter.size_hint());
    /// let _ = iter.next();
    /// assert_eq!((2, Some(2)), iter.size_hint());
    /// ```
    ///
    /// A more complex example:
    ///
    /// ```
    /// // The even numbers in the range of zero to nine.
    /// let iter = (0..10).filter(|x| x % 2 == 0);
    ///
    /// // We might iterate from zero to ten times. Knowing that it's five
    /// // exactly wouldn't be possible without executing filter().
    /// assert_eq!((0, Some(10)), iter.size_hint());
    ///
    /// // Let's add five more numbers with chain()
    /// let iter = (0..10).filter(|x| x % 2 == 0).chain(15..20);
    ///
    /// // now both bounds are increased by five
    /// assert_eq!((5, Some(15)), iter.size_hint());
    /// ```
    ///
    /// Returning `None` for an upper bound:
    ///
    /// ```
    /// // an infinite iterator has no upper bound
    /// // and the maximum possible lower bound
    /// let iter = 0..;
    ///
    /// assert_eq!((usize::MAX, None), iter.size_hint());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    /// Consumes the iterator, counting the number of iterations and returning it.
    ///
    /// This method will call [`next`] repeatedly until [`None`] is encountered,
    /// returning the number of times it saw [`Some`]. Note that [`next`] has to be
    /// called at least once even if the iterator does not have any elements.
    ///
    /// [`next`]: Iterator::next
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so counting elements of
    /// an iterator with more than [`usize::MAX`] elements either produces the
    /// wrong result or panics. If overflow checks are enabled, a panic is
    /// guaranteed.
    ///
    /// # Panics
    ///
    /// This function might panic if the iterator has more than [`usize::MAX`]
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().count(), 3);
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().count(), 5);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.fold(
            0,
            #[rustc_inherit_overflow_checks]
            |count, _| count + 1,
        )
    }

    /// Consumes the iterator, returning the last element.
    ///
    /// This method will evaluate the iterator until it returns [`None`]. While
    /// doing so, it keeps track of the current element. After [`None`] is
    /// returned, `last()` will then return the last element it saw.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.into_iter().last(), Some(3));
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.into_iter().last(), Some(5));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        #[inline]
        fn some<T>(_: Option<T>, x: T) -> Option<T> {
            Some(x)
        }

        self.fold(None, some)
    }

    /// Advances the iterator by `n` elements.
    ///
    /// This method will eagerly skip `n` elements by calling [`next`] up to `n`
    /// times until [`None`] is encountered.
    ///
    /// `advance_by(n)` will return `Ok(())` if the iterator successfully advances by
    /// `n` elements, or a `Err(NonZero<usize>)` with value `k` if [`None`] is encountered,
    /// where `k` is remaining number of steps that could not be advanced because the iterator ran out.
    /// If `self` is empty and `n` is non-zero, then this returns `Err(n)`.
    /// Otherwise, `k` is always less than `n`.
    ///
    /// Calling `advance_by(0)` can do meaningful work, for example [`Flatten`]
    /// can advance its outer iterator until it finds an inner iterator that is not empty, which
    /// then often allows it to return a more accurate `size_hint()` than in its initial state.
    ///
    /// [`Flatten`]: crate::iter::Flatten
    /// [`next`]: Iterator::next
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_advance_by)]
    ///
    /// use std::num::NonZero;
    ///
    /// let a = [1, 2, 3, 4];
    /// let mut iter = a.into_iter();
    ///
    /// assert_eq!(iter.advance_by(2), Ok(()));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.advance_by(0), Ok(()));
    /// assert_eq!(iter.advance_by(100), Err(NonZero::new(99).unwrap())); // only `4` was skipped
    /// ```
    #[inline]
    #[unstable(feature = "iter_advance_by", reason = "recently added", issue = "77404")]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        for i in 0..n {
            if self.next().is_none() {
                // SAFETY: `i` is always less than `n`.
                return Err(unsafe { NonZero::new_unchecked(n - i) });
            }
        }
        Ok(())
    }

    /// Returns the `n`th element of the iterator.
    ///
    /// Like most indexing operations, the count starts from zero, so `nth(0)`
    /// returns the first value, `nth(1)` the second, and so on.
    ///
    /// Note that all preceding elements, as well as the returned element, will be
    /// consumed from the iterator. That means that the preceding elements will be
    /// discarded, and also that calling `nth(0)` multiple times on the same iterator
    /// will return different elements.
    ///
    /// `nth()` will return [`None`] if `n` is greater than or equal to the length of the
    /// iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.into_iter().nth(1), Some(2));
    /// ```
    ///
    /// Calling `nth()` multiple times doesn't rewind the iterator:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// assert_eq!(iter.nth(1), Some(2));
    /// assert_eq!(iter.nth(1), None);
    /// ```
    ///
    /// Returning `None` if there are less than `n + 1` elements:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.into_iter().nth(10), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.advance_by(n).ok()?;
        self.next()
    }

    /// Creates an iterator starting at the same point, but stepping by
    /// the given amount at each iteration.
    ///
    /// Note 1: The first element of the iterator will always be returned,
    /// regardless of the step given.
    ///
    /// Note 2: The time at which ignored elements are pulled is not fixed.
    /// `StepBy` behaves like the sequence `self.next()`, `self.nth(step-1)`,
    /// `self.nth(step-1)`, …, but is also free to behave like the sequence
    /// `advance_n_and_return_first(&mut self, step)`,
    /// `advance_n_and_return_first(&mut self, step)`, …
    /// Which way is used may change for some iterators for performance reasons.
    /// The second way will advance the iterator earlier and may consume more items.
    ///
    /// `advance_n_and_return_first` is the equivalent of:
    /// ```
    /// fn advance_n_and_return_first<I>(iter: &mut I, n: usize) -> Option<I::Item>
    /// where
    ///     I: Iterator,
    /// {
    ///     let next = iter.next();
    ///     if n > 1 {
    ///         iter.nth(n - 2);
    ///     }
    ///     next
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The method will panic if the given step is `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [0, 1, 2, 3, 4, 5];
    /// let mut iter = a.into_iter().step_by(2);
    ///
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "iterator_step_by", since = "1.28.0")]
    fn step_by(self, step: usize) -> StepBy<Self>
    where
        Self: Sized,
    {
        StepBy::new(self, step)
    }

    /// Takes two iterators and creates a new iterator over both in sequence.
    ///
    /// `chain()` will return a new iterator which will first iterate over
    /// values from the first iterator and then over values from the second
    /// iterator.
    ///
    /// In other words, it links two iterators together, in a chain. 🔗
    ///
    /// [`once`] is commonly used to adapt a single value into a chain of
    /// other kinds of iteration.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s1 = "abc".chars();
    /// let s2 = "def".chars();
    ///
    /// let mut iter = s1.chain(s2);
    ///
    /// assert_eq!(iter.next(), Some('a'));
    /// assert_eq!(iter.next(), Some('b'));
    /// assert_eq!(iter.next(), Some('c'));
    /// assert_eq!(iter.next(), Some('d'));
    /// assert_eq!(iter.next(), Some('e'));
    /// assert_eq!(iter.next(), Some('f'));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Since the argument to `chain()` uses [`IntoIterator`], we can pass
    /// anything that can be converted into an [`Iterator`], not just an
    /// [`Iterator`] itself. For example, arrays (`[T]`) implement
    /// [`IntoIterator`], and so can be passed to `chain()` directly:
    ///
    /// ```
    /// let a1 = [1, 2, 3];
    /// let a2 = [4, 5, 6];
    ///
    /// let mut iter = a1.into_iter().chain(a2);
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// If you work with Windows API, you may wish to convert [`OsStr`] to `Vec<u16>`:
    ///
    /// ```
    /// #[cfg(windows)]
    /// fn os_str_to_utf16(s: &std::ffi::OsStr) -> Vec<u16> {
    ///     use std::os::windows::ffi::OsStrExt;
    ///     s.encode_wide().chain(std::iter::once(0)).collect()
    /// }
    /// ```
    ///
    /// [`once`]: crate::iter::once
    /// [`OsStr`]: ../../std/ffi/struct.OsStr.html
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chain<U>(self, other: U) -> Chain<Self, U::IntoIter>
    where
        Self: Sized,
        U: IntoIterator<Item = Self::Item>,
    {
        Chain::new(self, other.into_iter())
    }

    /// 'Zips up' two iterators into a single iterator of pairs.
    ///
    /// `zip()` returns a new iterator that will iterate over two other
    /// iterators, returning a tuple where the first element comes from the
    /// first iterator, and the second element comes from the second iterator.
    ///
    /// In other words, it zips two iterators together, into a single one.
    ///
    /// If either iterator returns [`None`], [`next`] from the zipped iterator
    /// will return [`None`].
    /// If the zipped iterator has no more elements to return then each further attempt to advance
    /// it will first try to advance the first iterator at most one time and if it still yielded an item
    /// try to advance the second iterator at most one time.
    ///
    /// To 'undo' the result of zipping up two iterators, see [`unzip`].
    ///
    /// [`unzip`]: Iterator::unzip
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s1 = "abc".chars();
    /// let s2 = "def".chars();
    ///
    /// let mut iter = s1.zip(s2);
    ///
    /// assert_eq!(iter.next(), Some(('a', 'd')));
    /// assert_eq!(iter.next(), Some(('b', 'e')));
    /// assert_eq!(iter.next(), Some(('c', 'f')));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Since the argument to `zip()` uses [`IntoIterator`], we can pass
    /// anything that can be converted into an [`Iterator`], not just an
    /// [`Iterator`] itself. For example, arrays (`[T]`) implement
    /// [`IntoIterator`], and so can be passed to `zip()` directly:
    ///
    /// ```
    /// let a1 = [1, 2, 3];
    /// let a2 = [4, 5, 6];
    ///
    /// let mut iter = a1.into_iter().zip(a2);
    ///
    /// assert_eq!(iter.next(), Some((1, 4)));
    /// assert_eq!(iter.next(), Some((2, 5)));
    /// assert_eq!(iter.next(), Some((3, 6)));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// `zip()` is often used to zip an infinite iterator to a finite one.
    /// This works because the finite iterator will eventually return [`None`],
    /// ending the zipper. Zipping with `(0..)` can look a lot like [`enumerate`]:
    ///
    /// ```
    /// let enumerate: Vec<_> = "foo".chars().enumerate().collect();
    ///
    /// let zipper: Vec<_> = (0..).zip("foo".chars()).collect();
    ///
    /// assert_eq!((0, 'f'), enumerate[0]);
    /// assert_eq!((0, 'f'), zipper[0]);
    ///
    /// assert_eq!((1, 'o'), enumerate[1]);
    /// assert_eq!((1, 'o'), zipper[1]);
    ///
    /// assert_eq!((2, 'o'), enumerate[2]);
    /// assert_eq!((2, 'o'), zipper[2]);
    /// ```
    ///
    /// If both iterators have roughly equivalent syntax, it may be more readable to use [`zip`]:
    ///
    /// ```
    /// use std::iter::zip;
    ///
    /// let a = [1, 2, 3];
    /// let b = [2, 3, 4];
    ///
    /// let mut zipped = zip(
    ///     a.into_iter().map(|x| x * 2).skip(1),
    ///     b.into_iter().map(|x| x * 2).skip(1),
    /// );
    ///
    /// assert_eq!(zipped.next(), Some((4, 6)));
    /// assert_eq!(zipped.next(), Some((6, 8)));
    /// assert_eq!(zipped.next(), None);
    /// ```
    ///
    /// compared to:
    ///
    /// ```
    /// # let a = [1, 2, 3];
    /// # let b = [2, 3, 4];
    /// #
    /// let mut zipped = a
    ///     .into_iter()
    ///     .map(|x| x * 2)
    ///     .skip(1)
    ///     .zip(b.into_iter().map(|x| x * 2).skip(1));
    /// #
    /// # assert_eq!(zipped.next(), Some((4, 6)));
    /// # assert_eq!(zipped.next(), Some((6, 8)));
    /// # assert_eq!(zipped.next(), None);
    /// ```
    ///
    /// [`enumerate`]: Iterator::enumerate
    /// [`next`]: Iterator::next
    /// [`zip`]: crate::iter::zip
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn zip<U>(self, other: U) -> Zip<Self, U::IntoIter>
    where
        Self: Sized,
        U: IntoIterator,
    {
        Zip::new(self, other.into_iter())
    }

    /// Creates a new iterator which places a copy of `separator` between adjacent
    /// items of the original iterator.
    ///
    /// In case `separator` does not implement [`Clone`] or needs to be
    /// computed every time, use [`intersperse_with`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_intersperse)]
    ///
    /// let mut a = [0, 1, 2].into_iter().intersperse(100);
    /// assert_eq!(a.next(), Some(0));   // The first element from `a`.
    /// assert_eq!(a.next(), Some(100)); // The separator.
    /// assert_eq!(a.next(), Some(1));   // The next element from `a`.
    /// assert_eq!(a.next(), Some(100)); // The separator.
    /// assert_eq!(a.next(), Some(2));   // The last element from `a`.
    /// assert_eq!(a.next(), None);       // The iterator is finished.
    /// ```
    ///
    /// `intersperse` can be very useful to join an iterator's items using a common element:
    /// ```
    /// #![feature(iter_intersperse)]
    ///
    /// let words = ["Hello", "World", "!"];
    /// let hello: String = words.into_iter().intersperse(" ").collect();
    /// assert_eq!(hello, "Hello World !");
    /// ```
    ///
    /// [`Clone`]: crate::clone::Clone
    /// [`intersperse_with`]: Iterator::intersperse_with
    #[inline]
    #[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
    fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        Intersperse::new(self, separator)
    }

    /// Creates a new iterator which places an item generated by `separator`
    /// between adjacent items of the original iterator.
    ///
    /// The closure will be called exactly once each time an item is placed
    /// between two adjacent items from the underlying iterator; specifically,
    /// the closure is not called if the underlying iterator yields less than
    /// two items and after the last item is yielded.
    ///
    /// If the iterator's item implements [`Clone`], it may be easier to use
    /// [`intersperse`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_intersperse)]
    ///
    /// #[derive(PartialEq, Debug)]
    /// struct NotClone(usize);
    ///
    /// let v = [NotClone(0), NotClone(1), NotClone(2)];
    /// let mut it = v.into_iter().intersperse_with(|| NotClone(99));
    ///
    /// assert_eq!(it.next(), Some(NotClone(0)));  // The first element from `v`.
    /// assert_eq!(it.next(), Some(NotClone(99))); // The separator.
    /// assert_eq!(it.next(), Some(NotClone(1)));  // The next element from `v`.
    /// assert_eq!(it.next(), Some(NotClone(99))); // The separator.
    /// assert_eq!(it.next(), Some(NotClone(2)));  // The last element from `v`.
    /// assert_eq!(it.next(), None);               // The iterator is finished.
    /// ```
    ///
    /// `intersperse_with` can be used in situations where the separator needs
    /// to be computed:
    /// ```
    /// #![feature(iter_intersperse)]
    ///
    /// let src = ["Hello", "to", "all", "people", "!!"].iter().copied();
    ///
    /// // The closure mutably borrows its context to generate an item.
    /// let mut happy_emojis = [" ❤️ ", " 😀 "].into_iter();
    /// let separator = || happy_emojis.next().unwrap_or(" 🦀 ");
    ///
    /// let result = src.intersperse_with(separator).collect::<String>();
    /// assert_eq!(result, "Hello ❤️ to 😀 all 🦀 people 🦀 !!");
    /// ```
    /// [`Clone`]: crate::clone::Clone
    /// [`intersperse`]: Iterator::intersperse
    #[inline]
    #[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
    fn intersperse_with<G>(self, separator: G) -> IntersperseWith<Self, G>
    where
        Self: Sized,
        G: FnMut() -> Self::Item,
    {
        IntersperseWith::new(self, separator)
    }

    /// Takes a closure and creates an iterator which calls that closure on each
    /// element.
    ///
    /// `map()` transforms one iterator into another, by means of its argument:
    /// something that implements [`FnMut`]. It produces a new iterator which
    /// calls this closure on each element of the original iterator.
    ///
    /// If you are good at thinking in types, you can think of `map()` like this:
    /// If you have an iterator that gives you elements of some type `A`, and
    /// you want an iterator of some other type `B`, you can use `map()`,
    /// passing a closure that takes an `A` and returns a `B`.
    ///
    /// `map()` is conceptually similar to a [`for`] loop. However, as `map()` is
    /// lazy, it is best used when you're already working with other iterators.
    /// If you're doing some sort of looping for a side effect, it's considered
    /// more idiomatic to use [`for`] than `map()`.
    ///
    /// [`for`]: ../../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter().map(|x| 2 * x);
    ///
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// If you're doing some sort of side effect, prefer [`for`] to `map()`:
    ///
    /// ```
    /// # #![allow(unused_must_use)]
    /// // don't do this:
    /// (0..5).map(|x| println!("{x}"));
    ///
    /// // it won't even execute, as it is lazy. Rust will warn you about this.
    ///
    /// // Instead, use a for-loop:
    /// for x in 0..5 {
    ///     println!("{x}");
    /// }
    /// ```
    #[rustc_diagnostic_item = "IteratorMap"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn map<B, F>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B,
    {
        Map::new(self, f)
    }

    /// Calls a closure on each element of an iterator.
    ///
    /// This is equivalent to using a [`for`] loop on the iterator, although
    /// `break` and `continue` are not possible from a closure. It's generally
    /// more idiomatic to use a `for` loop, but `for_each` may be more legible
    /// when processing items at the end of longer iterator chains. In some
    /// cases `for_each` may also be faster than a loop, because it will use
    /// internal iteration on adapters like `Chain`.
    ///
    /// [`for`]: ../../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::sync::mpsc::channel;
    ///
    /// let (tx, rx) = channel();
    /// (0..5).map(|x| x * 2 + 1)
    ///       .for_each(move |x| tx.send(x).unwrap());
    ///
    /// let v: Vec<_> = rx.iter().collect();
    /// assert_eq!(v, vec![1, 3, 5, 7, 9]);
    /// ```
    ///
    /// For such a small example, a `for` loop may be cleaner, but `for_each`
    /// might be preferable to keep a functional style with longer iterators:
    ///
    /// ```
    /// (0..5).flat_map(|x| x * 100 .. x * 110)
    ///       .enumerate()
    ///       .filter(|&(i, x)| (i + x) % 3 == 0)
    ///       .for_each(|(i, x)| println!("{i}:{x}"));
    /// ```
    #[inline]
    #[stable(feature = "iterator_for_each", since = "1.21.0")]
    fn for_each<F>(self, f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        #[inline]
        fn call<T>(mut f: impl FnMut(T)) -> impl FnMut((), T) {
            move |(), item| f(item)
        }

        self.fold((), call(f));
    }

    /// Creates an iterator which uses a closure to determine if an element
    /// should be yielded.
    ///
    /// Given an element the closure must return `true` or `false`. The returned
    /// iterator will yield only the elements for which the closure returns
    /// `true`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [0i32, 1, 2];
    ///
    /// let mut iter = a.into_iter().filter(|x| x.is_positive());
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the closure passed to `filter()` takes a reference, and many
    /// iterators iterate over references, this leads to a possibly confusing
    /// situation, where the type of the closure is a double reference:
    ///
    /// ```
    /// let s = &[0, 1, 2];
    ///
    /// let mut iter = s.iter().filter(|x| **x > 1); // needs two *s!
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// It's common to instead use destructuring on the argument to strip away one:
    ///
    /// ```
    /// let s = &[0, 1, 2];
    ///
    /// let mut iter = s.iter().filter(|&x| *x > 1); // both & and *
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// or both:
    ///
    /// ```
    /// let s = &[0, 1, 2];
    ///
    /// let mut iter = s.iter().filter(|&&x| x > 1); // two &s
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// of these layers.
    ///
    /// Note that `iter.filter(f).next()` is equivalent to `iter.find(f)`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "iter_filter"]
    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        Filter::new(self, predicate)
    }

    /// Creates an iterator that both filters and maps.
    ///
    /// The returned iterator yields only the `value`s for which the supplied
    /// closure returns `Some(value)`.
    ///
    /// `filter_map` can be used to make chains of [`filter`] and [`map`] more
    /// concise. The example below shows how a `map().filter().map()` can be
    /// shortened to a single call to `filter_map`.
    ///
    /// [`filter`]: Iterator::filter
    /// [`map`]: Iterator::map
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = ["1", "two", "NaN", "four", "5"];
    ///
    /// let mut iter = a.iter().filter_map(|s| s.parse().ok());
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Here's the same example, but with [`filter`] and [`map`]:
    ///
    /// ```
    /// let a = ["1", "two", "NaN", "four", "5"];
    /// let mut iter = a.iter().map(|s| s.parse()).filter(|s| s.is_ok()).map(|s| s.unwrap());
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<B>,
    {
        FilterMap::new(self, f)
    }

    /// Creates an iterator which gives the current iteration count as well as
    /// the next value.
    ///
    /// The iterator returned yields pairs `(i, val)`, where `i` is the
    /// current index of iteration and `val` is the value returned by the
    /// iterator.
    ///
    /// `enumerate()` keeps its count as a [`usize`]. If you want to count by a
    /// different sized integer, the [`zip`] function provides similar
    /// functionality.
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so enumerating more than
    /// [`usize::MAX`] elements either produces the wrong result or panics. If
    /// overflow checks are enabled, a panic is guaranteed.
    ///
    /// # Panics
    ///
    /// The returned iterator might panic if the to-be-returned index would
    /// overflow a [`usize`].
    ///
    /// [`zip`]: Iterator::zip
    ///
    /// # Examples
    ///
    /// ```
    /// let a = ['a', 'b', 'c'];
    ///
    /// let mut iter = a.into_iter().enumerate();
    ///
    /// assert_eq!(iter.next(), Some((0, 'a')));
    /// assert_eq!(iter.next(), Some((1, 'b')));
    /// assert_eq!(iter.next(), Some((2, 'c')));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "enumerate_method"]
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
    {
        Enumerate::new(self)
    }

    /// Creates an iterator which can use the [`peek`] and [`peek_mut`] methods
    /// to look at the next element of the iterator without consuming it. See
    /// their documentation for more information.
    ///
    /// Note that the underlying iterator is still advanced when [`peek`] or
    /// [`peek_mut`] are called for the first time: In order to retrieve the
    /// next element, [`next`] is called on the underlying iterator, hence any
    /// side effects (i.e. anything other than fetching the next value) of
    /// the [`next`] method will occur.
    ///
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let xs = [1, 2, 3];
    ///
    /// let mut iter = xs.into_iter().peekable();
    ///
    /// // peek() lets us see into the future
    /// assert_eq!(iter.peek(), Some(&1));
    /// assert_eq!(iter.next(), Some(1));
    ///
    /// assert_eq!(iter.next(), Some(2));
    ///
    /// // we can peek() multiple times, the iterator won't advance
    /// assert_eq!(iter.peek(), Some(&3));
    /// assert_eq!(iter.peek(), Some(&3));
    ///
    /// assert_eq!(iter.next(), Some(3));
    ///
    /// // after the iterator is finished, so is peek()
    /// assert_eq!(iter.peek(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Using [`peek_mut`] to mutate the next item without advancing the
    /// iterator:
    ///
    /// ```
    /// let xs = [1, 2, 3];
    ///
    /// let mut iter = xs.into_iter().peekable();
    ///
    /// // `peek_mut()` lets us see into the future
    /// assert_eq!(iter.peek_mut(), Some(&mut 1));
    /// assert_eq!(iter.peek_mut(), Some(&mut 1));
    /// assert_eq!(iter.next(), Some(1));
    ///
    /// if let Some(p) = iter.peek_mut() {
    ///     assert_eq!(*p, 2);
    ///     // put a value into the iterator
    ///     *p = 1000;
    /// }
    ///
    /// // The value reappears as the iterator continues
    /// assert_eq!(iter.collect::<Vec<_>>(), vec![1000, 3]);
    /// ```
    /// [`peek`]: Peekable::peek
    /// [`peek_mut`]: Peekable::peek_mut
    /// [`next`]: Iterator::next
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn peekable(self) -> Peekable<Self>
    where
        Self: Sized,
    {
        Peekable::new(self)
    }

    /// Creates an iterator that [`skip`]s elements based on a predicate.
    ///
    /// [`skip`]: Iterator::skip
    ///
    /// `skip_while()` takes a closure as an argument. It will call this
    /// closure on each element of the iterator, and ignore elements
    /// until it returns `false`.
    ///
    /// After `false` is returned, `skip_while()`'s job is over, and the
    /// rest of the elements are yielded.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [-1i32, 0, 1];
    ///
    /// let mut iter = a.into_iter().skip_while(|x| x.is_negative());
    ///
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the closure passed to `skip_while()` takes a reference, and many
    /// iterators iterate over references, this leads to a possibly confusing
    /// situation, where the type of the closure argument is a double reference:
    ///
    /// ```
    /// let s = &[-1, 0, 1];
    ///
    /// let mut iter = s.iter().skip_while(|x| **x < 0); // need two *s!
    ///
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Stopping after an initial `false`:
    ///
    /// ```
    /// let a = [-1, 0, 1, -2];
    ///
    /// let mut iter = a.into_iter().skip_while(|&x| x < 0);
    ///
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    ///
    /// // while this would have been false, since we already got a false,
    /// // skip_while() isn't used any more
    /// assert_eq!(iter.next(), Some(-2));
    ///
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[doc(alias = "drop_while")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        SkipWhile::new(self, predicate)
    }

    /// Creates an iterator that yields elements based on a predicate.
    ///
    /// `take_while()` takes a closure as an argument. It will call this
    /// closure on each element of the iterator, and yield elements
    /// while it returns `true`.
    ///
    /// After `false` is returned, `take_while()`'s job is over, and the
    /// rest of the elements are ignored.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [-1i32, 0, 1];
    ///
    /// let mut iter = a.into_iter().take_while(|x| x.is_negative());
    ///
    /// assert_eq!(iter.next(), Some(-1));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the closure passed to `take_while()` takes a reference, and many
    /// iterators iterate over references, this leads to a possibly confusing
    /// situation, where the type of the closure is a double reference:
    ///
    /// ```
    /// let s = &[-1, 0, 1];
    ///
    /// let mut iter = s.iter().take_while(|x| **x < 0); // need two *s!
    ///
    /// assert_eq!(iter.next(), Some(&-1));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Stopping after an initial `false`:
    ///
    /// ```
    /// let a = [-1, 0, 1, -2];
    ///
    /// let mut iter = a.into_iter().take_while(|&x| x < 0);
    ///
    /// assert_eq!(iter.next(), Some(-1));
    ///
    /// // We have more elements that are less than zero, but since we already
    /// // got a false, take_while() ignores the remaining elements.
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because `take_while()` needs to look at the value in order to see if it
    /// should be included or not, consuming iterators will see that it is
    /// removed:
    ///
    /// ```
    /// let a = [1, 2, 3, 4];
    /// let mut iter = a.into_iter();
    ///
    /// let result: Vec<i32> = iter.by_ref().take_while(|&n| n != 3).collect();
    ///
    /// assert_eq!(result, [1, 2]);
    ///
    /// let result: Vec<i32> = iter.collect();
    ///
    /// assert_eq!(result, [4]);
    /// ```
    ///
    /// The `3` is no longer there, because it was consumed in order to see if
    /// the iteration should stop, but wasn't placed back into the iterator.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        TakeWhile::new(self, predicate)
    }

    /// Creates an iterator that both yields elements based on a predicate and maps.
    ///
    /// `map_while()` takes a closure as an argument. It will call this
    /// closure on each element of the iterator, and yield elements
    /// while it returns [`Some(_)`][`Some`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [-1i32, 4, 0, 1];
    ///
    /// let mut iter = a.into_iter().map_while(|x| 16i32.checked_div(x));
    ///
    /// assert_eq!(iter.next(), Some(-16));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Here's the same example, but with [`take_while`] and [`map`]:
    ///
    /// [`take_while`]: Iterator::take_while
    /// [`map`]: Iterator::map
    ///
    /// ```
    /// let a = [-1i32, 4, 0, 1];
    ///
    /// let mut iter = a.into_iter()
    ///                 .map(|x| 16i32.checked_div(x))
    ///                 .take_while(|x| x.is_some())
    ///                 .map(|x| x.unwrap());
    ///
    /// assert_eq!(iter.next(), Some(-16));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Stopping after an initial [`None`]:
    ///
    /// ```
    /// let a = [0, 1, 2, -3, 4, 5, -6];
    ///
    /// let iter = a.into_iter().map_while(|x| u32::try_from(x).ok());
    /// let vec: Vec<_> = iter.collect();
    ///
    /// // We have more elements that could fit in u32 (such as 4, 5), but `map_while` returned `None` for `-3`
    /// // (as the `predicate` returned `None`) and `collect` stops at the first `None` encountered.
    /// assert_eq!(vec, [0, 1, 2]);
    /// ```
    ///
    /// Because `map_while()` needs to look at the value in order to see if it
    /// should be included or not, consuming iterators will see that it is
    /// removed:
    ///
    /// ```
    /// let a = [1, 2, -3, 4];
    /// let mut iter = a.into_iter();
    ///
    /// let result: Vec<u32> = iter.by_ref()
    ///                            .map_while(|n| u32::try_from(n).ok())
    ///                            .collect();
    ///
    /// assert_eq!(result, [1, 2]);
    ///
    /// let result: Vec<i32> = iter.collect();
    ///
    /// assert_eq!(result, [4]);
    /// ```
    ///
    /// The `-3` is no longer there, because it was consumed in order to see if
    /// the iteration should stop, but wasn't placed back into the iterator.
    ///
    /// Note that unlike [`take_while`] this iterator is **not** fused.
    /// It is also not specified what this iterator returns after the first [`None`] is returned.
    /// If you need a fused iterator, use [`fuse`].
    ///
    /// [`fuse`]: Iterator::fuse
    #[inline]
    #[stable(feature = "iter_map_while", since = "1.57.0")]
    fn map_while<B, P>(self, predicate: P) -> MapWhile<Self, P>
    where
        Self: Sized,
        P: FnMut(Self::Item) -> Option<B>,
    {
        MapWhile::new(self, predicate)
    }

    /// Creates an iterator that skips the first `n` elements.
    ///
    /// `skip(n)` skips elements until `n` elements are skipped or the end of the
    /// iterator is reached (whichever happens first). After that, all the remaining
    /// elements are yielded. In particular, if the original iterator is too short,
    /// then the returned iterator is empty.
    ///
    /// Rather than overriding this method directly, instead override the `nth` method.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter().skip(2);
    ///
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn skip(self, n: usize) -> Skip<Self>
    where
        Self: Sized,
    {
        Skip::new(self, n)
    }

    /// Creates an iterator that yields the first `n` elements, or fewer
    /// if the underlying iterator ends sooner.
    ///
    /// `take(n)` yields elements until `n` elements are yielded or the end of
    /// the iterator is reached (whichever happens first).
    /// The returned iterator is a prefix of length `n` if the original iterator
    /// contains at least `n` elements, otherwise it contains all of the
    /// (fewer than `n`) elements of the original iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter().take(2);
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// `take()` is often used with an infinite iterator, to make it finite:
    ///
    /// ```
    /// let mut iter = (0..).take(3);
    ///
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// If less than `n` elements are available,
    /// `take` will limit itself to the size of the underlying iterator:
    ///
    /// ```
    /// let v = [1, 2];
    /// let mut iter = v.into_iter().take(5);
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Use [`by_ref`] to take from the iterator without consuming it, and then
    /// continue using the original iterator:
    ///
    /// ```
    /// let mut words = ["hello", "world", "of", "Rust"].into_iter();
    ///
    /// // Take the first two words.
    /// let hello_world: Vec<_> = words.by_ref().take(2).collect();
    /// assert_eq!(hello_world, vec!["hello", "world"]);
    ///
    /// // Collect the rest of the words.
    /// // We can only do this because we used `by_ref` earlier.
    /// let of_rust: Vec<_> = words.collect();
    /// assert_eq!(of_rust, vec!["of", "Rust"]);
    /// ```
    ///
    /// [`by_ref`]: Iterator::by_ref
    #[doc(alias = "limit")]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take(self, n: usize) -> Take<Self>
    where
        Self: Sized,
    {
        Take::new(self, n)
    }

    /// An iterator adapter which, like [`fold`], holds internal state, but
    /// unlike [`fold`], produces a new iterator.
    ///
    /// [`fold`]: Iterator::fold
    ///
    /// `scan()` takes two arguments: an initial value which seeds the internal
    /// state, and a closure with two arguments, the first being a mutable
    /// reference to the internal state and the second an iterator element.
    /// The closure can assign to the internal state to share state between
    /// iterations.
    ///
    /// On iteration, the closure will be applied to each element of the
    /// iterator and the return value from the closure, an [`Option`], is
    /// returned by the `next` method. Thus the closure can return
    /// `Some(value)` to yield `value`, or `None` to end the iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4];
    ///
    /// let mut iter = a.into_iter().scan(1, |state, x| {
    ///     // each iteration, we'll multiply the state by the element ...
    ///     *state = *state * x;
    ///
    ///     // ... and terminate if the state exceeds 6
    ///     if *state > 6 {
    ///         return None;
    ///     }
    ///     // ... else yield the negation of the state
    ///     Some(-*state)
    /// });
    ///
    /// assert_eq!(iter.next(), Some(-1));
    /// assert_eq!(iter.next(), Some(-2));
    /// assert_eq!(iter.next(), Some(-6));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
    where
        Self: Sized,
        F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        Scan::new(self, initial_state, f)
    }

    /// Creates an iterator that works like map, but flattens nested structure.
    ///
    /// The [`map`] adapter is very useful, but only when the closure
    /// argument produces values. If it produces an iterator instead, there's
    /// an extra layer of indirection. `flat_map()` will remove this extra layer
    /// on its own.
    ///
    /// You can think of `flat_map(f)` as the semantic equivalent
    /// of [`map`]ping, and then [`flatten`]ing as in `map(f).flatten()`.
    ///
    /// Another way of thinking about `flat_map()`: [`map`]'s closure returns
    /// one item for each element, and `flat_map()`'s closure returns an
    /// iterator for each element.
    ///
    /// [`map`]: Iterator::map
    /// [`flatten`]: Iterator::flatten
    ///
    /// # Examples
    ///
    /// ```
    /// let words = ["alpha", "beta", "gamma"];
    ///
    /// // chars() returns an iterator
    /// let merged: String = words.iter()
    ///                           .flat_map(|s| s.chars())
    ///                           .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F>
    where
        Self: Sized,
        U: IntoIterator,
        F: FnMut(Self::Item) -> U,
    {
        FlatMap::new(self, f)
    }

    /// Creates an iterator that flattens nested structure.
    ///
    /// This is useful when you have an iterator of iterators or an iterator of
    /// things that can be turned into iterators and you want to remove one
    /// level of indirection.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let data = vec![vec![1, 2, 3, 4], vec![5, 6]];
    /// let flattened: Vec<_> = data.into_iter().flatten().collect();
    /// assert_eq!(flattened, [1, 2, 3, 4, 5, 6]);
    /// ```
    ///
    /// Mapping and then flattening:
    ///
    /// ```
    /// let words = ["alpha", "beta", "gamma"];
    ///
    /// // chars() returns an iterator
    /// let merged: String = words.iter()
    ///                           .map(|s| s.chars())
    ///                           .flatten()
    ///                           .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    ///
    /// You can also rewrite this in terms of [`flat_map()`], which is preferable
    /// in this case since it conveys intent more clearly:
    ///
    /// ```
    /// let words = ["alpha", "beta", "gamma"];
    ///
    /// // chars() returns an iterator
    /// let merged: String = words.iter()
    ///                           .flat_map(|s| s.chars())
    ///                           .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    ///
    /// Flattening works on any `IntoIterator` type, including `Option` and `Result`:
    ///
    /// ```
    /// let options = vec![Some(123), Some(321), None, Some(231)];
    /// let flattened_options: Vec<_> = options.into_iter().flatten().collect();
    /// assert_eq!(flattened_options, [123, 321, 231]);
    ///
    /// let results = vec![Ok(123), Ok(321), Err(456), Ok(231)];
    /// let flattened_results: Vec<_> = results.into_iter().flatten().collect();
    /// assert_eq!(flattened_results, [123, 321, 231]);
    /// ```
    ///
    /// Flattening only removes one level of nesting at a time:
    ///
    /// ```
    /// let d3 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    ///
    /// let d2: Vec<_> = d3.into_iter().flatten().collect();
    /// assert_eq!(d2, [[1, 2], [3, 4], [5, 6], [7, 8]]);
    ///
    /// let d1: Vec<_> = d3.into_iter().flatten().flatten().collect();
    /// assert_eq!(d1, [1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// Here we see that `flatten()` does not perform a "deep" flatten.
    /// Instead, only one level of nesting is removed. That is, if you
    /// `flatten()` a three-dimensional array, the result will be
    /// two-dimensional and not one-dimensional. To get a one-dimensional
    /// structure, you have to `flatten()` again.
    ///
    /// [`flat_map()`]: Iterator::flat_map
    #[inline]
    #[stable(feature = "iterator_flatten", since = "1.29.0")]
    fn flatten(self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: IntoIterator,
    {
        Flatten::new(self)
    }

    /// Calls the given function `f` for each contiguous window of size `N` over
    /// `self` and returns an iterator over the outputs of `f`. Like [`slice::windows()`],
    /// the windows during mapping overlap as well.
    ///
    /// In the following example, the closure is called three times with the
    /// arguments `&['a', 'b']`, `&['b', 'c']` and `&['c', 'd']` respectively.
    ///
    /// ```
    /// #![feature(iter_map_windows)]
    ///
    /// let strings = "abcd".chars()
    ///     .map_windows(|[x, y]| format!("{}+{}", x, y))
    ///     .collect::<Vec<String>>();
    ///
    /// assert_eq!(strings, vec!["a+b", "b+c", "c+d"]);
    /// ```
    ///
    /// Note that the const parameter `N` is usually inferred by the
    /// destructured argument in the closure.
    ///
    /// The returned iterator yields 𝑘 − `N` + 1 items (where 𝑘 is the number of
    /// items yielded by `self`). If 𝑘 is less than `N`, this method yields an
    /// empty iterator.
    ///
    /// The returned iterator implements [`FusedIterator`], because once `self`
    /// returns `None`, even if it returns a `Some(T)` again in the next iterations,
    /// we cannot put it into a contiguous array buffer, and thus the returned iterator
    /// should be fused.
    ///
    /// [`slice::windows()`]: slice::windows
    /// [`FusedIterator`]: crate::iter::FusedIterator
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero. This check will most probably get changed to a
    /// compile time error before this method gets stabilized.
    ///
    /// ```should_panic
    /// #![feature(iter_map_windows)]
    ///
    /// let iter = std::iter::repeat(0).map_windows(|&[]| ());
    /// ```
    ///
    /// # Examples
    ///
    /// Building the sums of neighboring numbers.
    ///
    /// ```
    /// #![feature(iter_map_windows)]
    ///
    /// let mut it = [1, 3, 8, 1].iter().map_windows(|&[a, b]| a + b);
    /// assert_eq!(it.next(), Some(4));  // 1 + 3
    /// assert_eq!(it.next(), Some(11)); // 3 + 8
    /// assert_eq!(it.next(), Some(9));  // 8 + 1
    /// assert_eq!(it.next(), None);
    /// ```
    ///
    /// Since the elements in the following example implement `Copy`, we can
    /// just copy the array and get an iterator over the windows.
    ///
    /// ```
    /// #![feature(iter_map_windows)]
    ///
    /// let mut it = "ferris".chars().map_windows(|w: &[_; 3]| *w);
    /// assert_eq!(it.next(), Some(['f', 'e', 'r']));
    /// assert_eq!(it.next(), Some(['e', 'r', 'r']));
    /// assert_eq!(it.next(), Some(['r', 'r', 'i']));
    /// assert_eq!(it.next(), Some(['r', 'i', 's']));
    /// assert_eq!(it.next(), None);
    /// ```
    ///
    /// You can also use this function to check the sortedness of an iterator.
    /// For the simple case, rather use [`Iterator::is_sorted`].
    ///
    /// ```
    /// #![feature(iter_map_windows)]
    ///
    /// let mut it = [0.5, 1.0, 3.5, 3.0, 8.5, 8.5, f32::NAN].iter()
    ///     .map_windows(|[a, b]| a <= b);
    ///
    /// assert_eq!(it.next(), Some(true));  // 0.5 <= 1.0
    /// assert_eq!(it.next(), Some(true));  // 1.0 <= 3.5
    /// assert_eq!(it.next(), Some(false)); // 3.5 <= 3.0
    /// assert_eq!(it.next(), Some(true));  // 3.0 <= 8.5
    /// assert_eq!(it.next(), Some(true));  // 8.5 <= 8.5
    /// assert_eq!(it.next(), Some(false)); // 8.5 <= NAN
    /// assert_eq!(it.next(), None);
    /// ```
    ///
    /// For non-fused iterators, they are fused after `map_windows`.
    ///
    /// ```
    /// #![feature(iter_map_windows)]
    ///
    /// #[derive(Default)]
    /// struct NonFusedIterator {
    ///     state: i32,
    /// }
    ///
    /// impl Iterator for NonFusedIterator {
    ///     type Item = i32;
    ///
    ///     fn next(&mut self) -> Option<i32> {
    ///         let val = self.state;
    ///         self.state = self.state + 1;
    ///
    ///         // yields `0..5` first, then only even numbers since `6..`.
    ///         if val < 5 || val % 2 == 0 {
    ///             Some(val)
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    ///
    ///
    /// let mut iter = NonFusedIterator::default();
    ///
    /// // yields 0..5 first.
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(4));
    /// // then we can see our iterator going back and forth
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), Some(8));
    /// assert_eq!(iter.next(), None);
    ///
    /// // however, with `.map_windows()`, it is fused.
    /// let mut iter = NonFusedIterator::default()
    ///     .map_windows(|arr: &[_; 2]| *arr);
    ///
    /// assert_eq!(iter.next(), Some([0, 1]));
    /// assert_eq!(iter.next(), Some([1, 2]));
    /// assert_eq!(iter.next(), Some([2, 3]));
    /// assert_eq!(iter.next(), Some([3, 4]));
    /// assert_eq!(iter.next(), None);
    ///
    /// // it will always return `None` after the first time.
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[unstable(feature = "iter_map_windows", reason = "recently added", issue = "87155")]
    fn map_windows<F, R, const N: usize>(self, f: F) -> MapWindows<Self, F, N>
    where
        Self: Sized,
        F: FnMut(&[Self::Item; N]) -> R,
    {
        MapWindows::new(self, f)
    }

    /// Creates an iterator which ends after the first [`None`].
    ///
    /// After an iterator returns [`None`], future calls may or may not yield
    /// [`Some(T)`] again. `fuse()` adapts an iterator, ensuring that after a
    /// [`None`] is given, it will always return [`None`] forever.
    ///
    /// Note that the [`Fuse`] wrapper is a no-op on iterators that implement
    /// the [`FusedIterator`] trait. `fuse()` may therefore behave incorrectly
    /// if the [`FusedIterator`] trait is improperly implemented.
    ///
    /// [`Some(T)`]: Some
    /// [`FusedIterator`]: crate::iter::FusedIterator
    ///
    /// # Examples
    ///
    /// ```
    /// // an iterator which alternates between Some and None
    /// struct Alternate {
    ///     state: i32,
    /// }
    ///
    /// impl Iterator for Alternate {
    ///     type Item = i32;
    ///
    ///     fn next(&mut self) -> Option<i32> {
    ///         let val = self.state;
    ///         self.state = self.state + 1;
    ///
    ///         // if it's even, Some(i32), else None
    ///         (val % 2 == 0).then_some(val)
    ///     }
    /// }
    ///
    /// let mut iter = Alternate { state: 0 };
    ///
    /// // we can see our iterator going back and forth
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    ///
    /// // however, once we fuse it...
    /// let mut iter = iter.fuse();
    ///
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), None);
    ///
    /// // it will always return `None` after the first time.
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fuse(self) -> Fuse<Self>
    where
        Self: Sized,
    {
        Fuse::new(self)
    }

    /// Does something with each element of an iterator, passing the value on.
    ///
    /// When using iterators, you'll often chain several of them together.
    /// While working on such code, you might want to check out what's
    /// happening at various parts in the pipeline. To do that, insert
    /// a call to `inspect()`.
    ///
    /// It's more common for `inspect()` to be used as a debugging tool than to
    /// exist in your final code, but applications may find it useful in certain
    /// situations when errors need to be logged before being discarded.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 4, 2, 3];
    ///
    /// // this iterator sequence is complex.
    /// let sum = a.iter()
    ///     .cloned()
    ///     .filter(|x| x % 2 == 0)
    ///     .fold(0, |sum, i| sum + i);
    ///
    /// println!("{sum}");
    ///
    /// // let's add some inspect() calls to investigate what's happening
    /// let sum = a.iter()
    ///     .cloned()
    ///     .inspect(|x| println!("about to filter: {x}"))
    ///     .filter(|x| x % 2 == 0)
    ///     .inspect(|x| println!("made it through filter: {x}"))
    ///     .fold(0, |sum, i| sum + i);
    ///
    /// println!("{sum}");
    /// ```
    ///
    /// This will print:
    ///
    /// ```text
    /// 6
    /// about to filter: 1
    /// about to filter: 4
    /// made it through filter: 4
    /// about to filter: 2
    /// made it through filter: 2
    /// about to filter: 3
    /// 6
    /// ```
    ///
    /// Logging errors before discarding them:
    ///
    /// ```
    /// let lines = ["1", "2", "a"];
    ///
    /// let sum: i32 = lines
    ///     .iter()
    ///     .map(|line| line.parse::<i32>())
    ///     .inspect(|num| {
    ///         if let Err(ref e) = *num {
    ///             println!("Parsing error: {e}");
    ///         }
    ///     })
    ///     .filter_map(Result::ok)
    ///     .sum();
    ///
    /// println!("Sum: {sum}");
    /// ```
    ///
    /// This will print:
    ///
    /// ```text
    /// Parsing error: invalid digit found in string
    /// Sum: 3
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        Self: Sized,
        F: FnMut(&Self::Item),
    {
        Inspect::new(self, f)
    }

    /// Creates a "by reference" adapter for this instance of `Iterator`.
    ///
    /// Consuming method calls (direct or indirect calls to `next`)
    /// on the "by reference" adapter will consume the original iterator,
    /// but ownership-taking methods (those with a `self` parameter)
    /// only take ownership of the "by reference" iterator.
    ///
    /// This is useful for applying ownership-taking methods
    /// (such as `take` in the example below)
    /// without giving up ownership of the original iterator,
    /// so you can use the original iterator afterwards.
    ///
    /// Uses [impl<I: Iterator + ?Sized> Iterator for &mut I { type Item = I::Item; ...}](https://doc.rust-lang.org/nightly/std/iter/trait.Iterator.html#impl-Iterator-for-%26mut+I).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut words = ["hello", "world", "of", "Rust"].into_iter();
    ///
    /// // Take the first two words.
    /// let hello_world: Vec<_> = words.by_ref().take(2).collect();
    /// assert_eq!(hello_world, vec!["hello", "world"]);
    ///
    /// // Collect the rest of the words.
    /// // We can only do this because we used `by_ref` earlier.
    /// let of_rust: Vec<_> = words.collect();
    /// assert_eq!(of_rust, vec!["of", "Rust"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn by_ref(&mut self) -> &mut Self
    where
        Self: Sized,
    {
        self
    }

    /// Transforms an iterator into a collection.
    ///
    /// `collect()` can take anything iterable, and turn it into a relevant
    /// collection. This is one of the more powerful methods in the standard
    /// library, used in a variety of contexts.
    ///
    /// The most basic pattern in which `collect()` is used is to turn one
    /// collection into another. You take a collection, call [`iter`] on it,
    /// do a bunch of transformations, and then `collect()` at the end.
    ///
    /// `collect()` can also create instances of types that are not typical
    /// collections. For example, a [`String`] can be built from [`char`]s,
    /// and an iterator of [`Result<T, E>`][`Result`] items can be collected
    /// into `Result<Collection<T>, E>`. See the examples below for more.
    ///
    /// Because `collect()` is so general, it can cause problems with type
    /// inference. As such, `collect()` is one of the few times you'll see
    /// the syntax affectionately known as the 'turbofish': `::<>`. This
    /// helps the inference algorithm understand specifically which collection
    /// you're trying to collect into.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let doubled: Vec<i32> = a.iter()
    ///                          .map(|x| x * 2)
    ///                          .collect();
    ///
    /// assert_eq!(vec![2, 4, 6], doubled);
    /// ```
    ///
    /// Note that we needed the `: Vec<i32>` on the left-hand side. This is because
    /// we could collect into, for example, a [`VecDeque<T>`] instead:
    ///
    /// [`VecDeque<T>`]: ../../std/collections/struct.VecDeque.html
    ///
    /// ```
    /// use std::collections::VecDeque;
    ///
    /// let a = [1, 2, 3];
    ///
    /// let doubled: VecDeque<i32> = a.iter().map(|x| x * 2).collect();
    ///
    /// assert_eq!(2, doubled[0]);
    /// assert_eq!(4, doubled[1]);
    /// assert_eq!(6, doubled[2]);
    /// ```
    ///
    /// Using the 'turbofish' instead of annotating `doubled`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let doubled = a.iter().map(|x| x * 2).collect::<Vec<i32>>();
    ///
    /// assert_eq!(vec![2, 4, 6], doubled);
    /// ```
    ///
    /// Because `collect()` only cares about what you're collecting into, you can
    /// still use a partial type hint, `_`, with the turbofish:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let doubled = a.iter().map(|x| x * 2).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![2, 4, 6], doubled);
    /// ```
    ///
    /// Using `collect()` to make a [`String`]:
    ///
    /// ```
    /// let chars = ['g', 'd', 'k', 'k', 'n'];
    ///
    /// let hello: String = chars.into_iter()
    ///     .map(|x| x as u8)
    ///     .map(|x| (x + 1) as char)
    ///     .collect();
    ///
    /// assert_eq!("hello", hello);
    /// ```
    ///
    /// If you have a list of [`Result<T, E>`][`Result`]s, you can use `collect()` to
    /// see if any of them failed:
    ///
    /// ```
    /// let results = [Ok(1), Err("nope"), Ok(3), Err("bad")];
    ///
    /// let result: Result<Vec<_>, &str> = results.into_iter().collect();
    ///
    /// // gives us the first error
    /// assert_eq!(Err("nope"), result);
    ///
    /// let results = [Ok(1), Ok(3)];
    ///
    /// let result: Result<Vec<_>, &str> = results.into_iter().collect();
    ///
    /// // gives us the list of answers
    /// assert_eq!(Ok(vec![1, 3]), result);
    /// ```
    ///
    /// [`iter`]: Iterator::next
    /// [`String`]: ../../std/string/struct.String.html
    /// [`char`]: type@char
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "if you really need to exhaust the iterator, consider `.for_each(drop)` instead"]
    #[rustc_diagnostic_item = "iterator_collect_fn"]
    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        // This is too aggressive to turn on for everything all the time, but PR#137908
        // accidentally noticed that some rustc iterators had malformed `size_hint`s,
        // so this will help catch such things in debug-assertions-std runners,
        // even if users won't actually ever see it.
        if cfg!(debug_assertions) {
            let hint = self.size_hint();
            assert!(hint.1.is_none_or(|high| high >= hint.0), "Malformed size_hint {hint:?}");
        }

        FromIterator::from_iter(self)
    }

    /// Fallibly transforms an iterator into a collection, short circuiting if
    /// a failure is encountered.
    ///
    /// `try_collect()` is a variation of [`collect()`][`collect`] that allows fallible
    /// conversions during collection. Its main use case is simplifying conversions from
    /// iterators yielding [`Option<T>`][`Option`] into `Option<Collection<T>>`, or similarly for other [`Try`]
    /// types (e.g. [`Result`]).
    ///
    /// Importantly, `try_collect()` doesn't require that the outer [`Try`] type also implements [`FromIterator`];
    /// only the inner type produced on `Try::Output` must implement it. Concretely,
    /// this means that collecting into `ControlFlow<_, Vec<i32>>` is valid because `Vec<i32>` implements
    /// [`FromIterator`], even though [`ControlFlow`] doesn't.
    ///
    /// Also, if a failure is encountered during `try_collect()`, the iterator is still valid and
    /// may continue to be used, in which case it will continue iterating starting after the element that
    /// triggered the failure. See the last example below for an example of how this works.
    ///
    /// # Examples
    /// Successfully collecting an iterator of `Option<i32>` into `Option<Vec<i32>>`:
    /// ```
    /// #![feature(iterator_try_collect)]
    ///
    /// let u = vec![Some(1), Some(2), Some(3)];
    /// let v = u.into_iter().try_collect::<Vec<i32>>();
    /// assert_eq!(v, Some(vec![1, 2, 3]));
    /// ```
    ///
    /// Failing to collect in the same way:
    /// ```
    /// #![feature(iterator_try_collect)]
    ///
    /// let u = vec![Some(1), Some(2), None, Some(3)];
    /// let v = u.into_iter().try_collect::<Vec<i32>>();
    /// assert_eq!(v, None);
    /// ```
    ///
    /// A similar example, but with `Result`:
    /// ```
    /// #![feature(iterator_try_collect)]
    ///
    /// let u: Vec<Result<i32, ()>> = vec![Ok(1), Ok(2), Ok(3)];
    /// let v = u.into_iter().try_collect::<Vec<i32>>();
    /// assert_eq!(v, Ok(vec![1, 2, 3]));
    ///
    /// let u = vec![Ok(1), Ok(2), Err(()), Ok(3)];
    /// let v = u.into_iter().try_collect::<Vec<i32>>();
    /// assert_eq!(v, Err(()));
    /// ```
    ///
    /// Finally, even [`ControlFlow`] works, despite the fact that it
    /// doesn't implement [`FromIterator`]. Note also that the iterator can
    /// continue to be used, even if a failure is encountered:
    ///
    /// ```
    /// #![feature(iterator_try_collect)]
    ///
    /// use core::ops::ControlFlow::{Break, Continue};
    ///
    /// let u = [Continue(1), Continue(2), Break(3), Continue(4), Continue(5)];
    /// let mut it = u.into_iter();
    ///
    /// let v = it.try_collect::<Vec<_>>();
    /// assert_eq!(v, Break(3));
    ///
    /// let v = it.try_collect::<Vec<_>>();
    /// assert_eq!(v, Continue(vec![4, 5]));
    /// ```
    ///
    /// [`collect`]: Iterator::collect
    #[inline]
    #[unstable(feature = "iterator_try_collect", issue = "94047")]
    fn try_collect<B>(&mut self) -> ChangeOutputType<Self::Item, B>
    where
        Self: Sized,
        Self::Item: Try<Residual: Residual<B>>,
        B: FromIterator<<Self::Item as Try>::Output>,
    {
        try_process(ByRefSized(self), |i| i.collect())
    }

    /// Collects all the items from an iterator into a collection.
    ///
    /// This method consumes the iterator and adds all its items to the
    /// passed collection. The collection is then returned, so the call chain
    /// can be continued.
    ///
    /// This is useful when you already have a collection and want to add
    /// the iterator items to it.
    ///
    /// This method is a convenience method to call [Extend::extend](trait.Extend.html),
    /// but instead of being called on a collection, it's called on an iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_collect_into)]
    ///
    /// let a = [1, 2, 3];
    /// let mut vec: Vec::<i32> = vec![0, 1];
    ///
    /// a.iter().map(|x| x * 2).collect_into(&mut vec);
    /// a.iter().map(|x| x * 10).collect_into(&mut vec);
    ///
    /// assert_eq!(vec, vec![0, 1, 2, 4, 6, 10, 20, 30]);
    /// ```
    ///
    /// `Vec` can have a manual set capacity to avoid reallocating it:
    ///
    /// ```
    /// #![feature(iter_collect_into)]
    ///
    /// let a = [1, 2, 3];
    /// let mut vec: Vec::<i32> = Vec::with_capacity(6);
    ///
    /// a.iter().map(|x| x * 2).collect_into(&mut vec);
    /// a.iter().map(|x| x * 10).collect_into(&mut vec);
    ///
    /// assert_eq!(6, vec.capacity());
    /// assert_eq!(vec, vec![2, 4, 6, 10, 20, 30]);
    /// ```
    ///
    /// The returned mutable reference can be used to continue the call chain:
    ///
    /// ```
    /// #![feature(iter_collect_into)]
    ///
    /// let a = [1, 2, 3];
    /// let mut vec: Vec::<i32> = Vec::with_capacity(6);
    ///
    /// let count = a.iter().collect_into(&mut vec).iter().count();
    ///
    /// assert_eq!(count, vec.len());
    /// assert_eq!(vec, vec![1, 2, 3]);
    ///
    /// let count = a.iter().collect_into(&mut vec).iter().count();
    ///
    /// assert_eq!(count, vec.len());
    /// assert_eq!(vec, vec![1, 2, 3, 1, 2, 3]);
    /// ```
    #[inline]
    #[unstable(feature = "iter_collect_into", reason = "new API", issue = "94780")]
    fn collect_into<E: Extend<Self::Item>>(self, collection: &mut E) -> &mut E
    where
        Self: Sized,
    {
        collection.extend(self);
        collection
    }

    /// Consumes an iterator, creating two collections from it.
    ///
    /// The predicate passed to `partition()` can return `true`, or `false`.
    /// `partition()` returns a pair, all of the elements for which it returned
    /// `true`, and all of the elements for which it returned `false`.
    ///
    /// See also [`is_partitioned()`] and [`partition_in_place()`].
    ///
    /// [`is_partitioned()`]: Iterator::is_partitioned
    /// [`partition_in_place()`]: Iterator::partition_in_place
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let (even, odd): (Vec<_>, Vec<_>) = a
    ///     .into_iter()
    ///     .partition(|n| n % 2 == 0);
    ///
    /// assert_eq!(even, [2]);
    /// assert_eq!(odd, [1, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn partition<B, F>(self, f: F) -> (B, B)
    where
        Self: Sized,
        B: Default + Extend<Self::Item>,
        F: FnMut(&Self::Item) -> bool,
    {
        #[inline]
        fn extend<'a, T, B: Extend<T>>(
            mut f: impl FnMut(&T) -> bool + 'a,
            left: &'a mut B,
            right: &'a mut B,
        ) -> impl FnMut((), T) + 'a {
            move |(), x| {
                if f(&x) {
                    left.extend_one(x);
                } else {
                    right.extend_one(x);
                }
            }
        }

        let mut left: B = Default::default();
        let mut right: B = Default::default();

        self.fold((), extend(f, &mut left, &mut right));

        (left, right)
    }

    /// Reorders the elements of this iterator *in-place* according to the given predicate,
    /// such that all those that return `true` precede all those that return `false`.
    /// Returns the number of `true` elements found.
    ///
    /// The relative order of partitioned items is not maintained.
    ///
    /// # Current implementation
    ///
    /// The current algorithm tries to find the first element for which the predicate evaluates
    /// to false and the last element for which it evaluates to true, and repeatedly swaps them.
    ///
    /// Time complexity: *O*(*n*)
    ///
    /// See also [`is_partitioned()`] and [`partition()`].
    ///
    /// [`is_partitioned()`]: Iterator::is_partitioned
    /// [`partition()`]: Iterator::partition
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_partition_in_place)]
    ///
    /// let mut a = [1, 2, 3, 4, 5, 6, 7];
    ///
    /// // Partition in-place between evens and odds
    /// let i = a.iter_mut().partition_in_place(|n| n % 2 == 0);
    ///
    /// assert_eq!(i, 3);
    /// assert!(a[..i].iter().all(|n| n % 2 == 0)); // evens
    /// assert!(a[i..].iter().all(|n| n % 2 == 1)); // odds
    /// ```
    #[unstable(feature = "iter_partition_in_place", reason = "new API", issue = "62543")]
    fn partition_in_place<'a, T: 'a, P>(mut self, ref mut predicate: P) -> usize
    where
        Self: Sized + DoubleEndedIterator<Item = &'a mut T>,
        P: FnMut(&T) -> bool,
    {
        // FIXME: should we worry about the count overflowing? The only way to have more than
        // `usize::MAX` mutable references is with ZSTs, which aren't useful to partition...

        // These closure "factory" functions exist to avoid genericity in `Self`.

        #[inline]
        fn is_false<'a, T>(
            predicate: &'a mut impl FnMut(&T) -> bool,
            true_count: &'a mut usize,
        ) -> impl FnMut(&&mut T) -> bool + 'a {
            move |x| {
                let p = predicate(&**x);
                *true_count += p as usize;
                !p
            }
        }

        #[inline]
        fn is_true<T>(predicate: &mut impl FnMut(&T) -> bool) -> impl FnMut(&&mut T) -> bool + '_ {
            move |x| predicate(&**x)
        }

        // Repeatedly find the first `false` and swap it with the last `true`.
        let mut true_count = 0;
        while let Some(head) = self.find(is_false(predicate, &mut true_count)) {
            if let Some(tail) = self.rfind(is_true(predicate)) {
                crate::mem::swap(head, tail);
                true_count += 1;
            } else {
                break;
            }
        }
        true_count
    }

    /// Checks if the elements of this iterator are partitioned according to the given predicate,
    /// such that all those that return `true` precede all those that return `false`.
    ///
    /// See also [`partition()`] and [`partition_in_place()`].
    ///
    /// [`partition()`]: Iterator::partition
    /// [`partition_in_place()`]: Iterator::partition_in_place
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_is_partitioned)]
    ///
    /// assert!("Iterator".chars().is_partitioned(char::is_uppercase));
    /// assert!(!"IntoIterator".chars().is_partitioned(char::is_uppercase));
    /// ```
    #[unstable(feature = "iter_is_partitioned", reason = "new API", issue = "62544")]
    fn is_partitioned<P>(mut self, mut predicate: P) -> bool
    where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        // Either all items test `true`, or the first clause stops at `false`
        // and we check that there are no more `true` items after that.
        self.all(&mut predicate) || !self.any(predicate)
    }

    /// An iterator method that applies a function as long as it returns
    /// successfully, producing a single, final value.
    ///
    /// `try_fold()` takes two arguments: an initial value, and a closure with
    /// two arguments: an 'accumulator', and an element. The closure either
    /// returns successfully, with the value that the accumulator should have
    /// for the next iteration, or it returns failure, with an error value that
    /// is propagated back to the caller immediately (short-circuiting).
    ///
    /// The initial value is the value the accumulator will have on the first
    /// call. If applying the closure succeeded against every element of the
    /// iterator, `try_fold()` returns the final accumulator as success.
    ///
    /// Folding is useful whenever you have a collection of something, and want
    /// to produce a single value from it.
    ///
    /// # Note to Implementors
    ///
    /// Several of the other (forward) methods have default implementations in
    /// terms of this one, so try to implement this explicitly if it can
    /// do something better than the default `for` loop implementation.
    ///
    /// In particular, try to have this call `try_fold()` on the internal parts
    /// from which this iterator is composed. If multiple calls are needed,
    /// the `?` operator may be convenient for chaining the accumulator value
    /// along, but beware any invariants that need to be upheld before those
    /// early returns. This is a `&mut self` method, so iteration needs to be
    /// resumable after hitting an error here.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// // the checked sum of all of the elements of the array
    /// let sum = a.into_iter().try_fold(0i8, |acc, x| acc.checked_add(x));
    ///
    /// assert_eq!(sum, Some(6));
    /// ```
    ///
    /// Short-circuiting:
    ///
    /// ```
    /// let a = [10, 20, 30, 100, 40, 50];
    /// let mut iter = a.into_iter();
    ///
    /// // This sum overflows when adding the 100 element
    /// let sum = iter.try_fold(0i8, |acc, x| acc.checked_add(x));
    /// assert_eq!(sum, None);
    ///
    /// // Because it short-circuited, the remaining elements are still
    /// // available through the iterator.
    /// assert_eq!(iter.len(), 2);
    /// assert_eq!(iter.next(), Some(40));
    /// ```
    ///
    /// While you cannot `break` from a closure, the [`ControlFlow`] type allows
    /// a similar idea:
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// let triangular = (1..30).try_fold(0_i8, |prev, x| {
    ///     if let Some(next) = prev.checked_add(x) {
    ///         ControlFlow::Continue(next)
    ///     } else {
    ///         ControlFlow::Break(prev)
    ///     }
    /// });
    /// assert_eq!(triangular, ControlFlow::Break(120));
    ///
    /// let triangular = (1..30).try_fold(0_u64, |prev, x| {
    ///     if let Some(next) = prev.checked_add(x) {
    ///         ControlFlow::Continue(next)
    ///     } else {
    ///         ControlFlow::Break(prev)
    ///     }
    /// });
    /// assert_eq!(triangular, ControlFlow::Continue(435));
    /// ```
    #[inline]
    #[stable(feature = "iterator_try_fold", since = "1.27.0")]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x)?;
        }
        try { accum }
    }

    /// An iterator method that applies a fallible function to each item in the
    /// iterator, stopping at the first error and returning that error.
    ///
    /// This can also be thought of as the fallible form of [`for_each()`]
    /// or as the stateless version of [`try_fold()`].
    ///
    /// [`for_each()`]: Iterator::for_each
    /// [`try_fold()`]: Iterator::try_fold
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::rename;
    /// use std::io::{stdout, Write};
    /// use std::path::Path;
    ///
    /// let data = ["no_tea.txt", "stale_bread.json", "torrential_rain.png"];
    ///
    /// let res = data.iter().try_for_each(|x| writeln!(stdout(), "{x}"));
    /// assert!(res.is_ok());
    ///
    /// let mut it = data.iter().cloned();
    /// let res = it.try_for_each(|x| rename(x, Path::new(x).with_extension("old")));
    /// assert!(res.is_err());
    /// // It short-circuited, so the remaining items are still in the iterator:
    /// assert_eq!(it.next(), Some("stale_bread.json"));
    /// ```
    ///
    /// The [`ControlFlow`] type can be used with this method for the situations
    /// in which you'd use `break` and `continue` in a normal loop:
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// let r = (2..100).try_for_each(|x| {
    ///     if 323 % x == 0 {
    ///         return ControlFlow::Break(x)
    ///     }
    ///
    ///     ControlFlow::Continue(())
    /// });
    /// assert_eq!(r, ControlFlow::Break(17));
    /// ```
    #[inline]
    #[stable(feature = "iterator_try_fold", since = "1.27.0")]
    fn try_for_each<F, R>(&mut self, f: F) -> R
    where
        Self: Sized,
        F: FnMut(Self::Item) -> R,
        R: Try<Output = ()>,
    {
        #[inline]
        fn call<T, R>(mut f: impl FnMut(T) -> R) -> impl FnMut((), T) -> R {
            move |(), x| f(x)
        }

        self.try_fold((), call(f))
    }

    /// Folds every element into an accumulator by applying an operation,
    /// returning the final result.
    ///
    /// `fold()` takes two arguments: an initial value, and a closure with two
    /// arguments: an 'accumulator', and an element. The closure returns the value that
    /// the accumulator should have for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first
    /// call.
    ///
    /// After applying this closure to every element of the iterator, `fold()`
    /// returns the accumulator.
    ///
    /// This operation is sometimes called 'reduce' or 'inject'.
    ///
    /// Folding is useful whenever you have a collection of something, and want
    /// to produce a single value from it.
    ///
    /// Note: `fold()`, and similar methods that traverse the entire iterator,
    /// might not terminate for infinite iterators, even on traits for which a
    /// result is determinable in finite time.
    ///
    /// Note: [`reduce()`] can be used to use the first element as the initial
    /// value, if the accumulator type and item type is the same.
    ///
    /// Note: `fold()` combines elements in a *left-associative* fashion. For associative
    /// operators like `+`, the order the elements are combined in is not important, but for non-associative
    /// operators like `-` the order will affect the final result.
    /// For a *right-associative* version of `fold()`, see [`DoubleEndedIterator::rfold()`].
    ///
    /// # Note to Implementors
    ///
    /// Several of the other (forward) methods have default implementations in
    /// terms of this one, so try to implement this explicitly if it can
    /// do something better than the default `for` loop implementation.
    ///
    /// In particular, try to have this call `fold()` on the internal parts
    /// from which this iterator is composed.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// // the sum of all of the elements of the array
    /// let sum = a.iter().fold(0, |acc, x| acc + x);
    ///
    /// assert_eq!(sum, 6);
    /// ```
    ///
    /// Let's walk through each step of the iteration here:
    ///
    /// | element | acc | x | result |
    /// |---------|-----|---|--------|
    /// |         | 0   |   |        |
    /// | 1       | 0   | 1 | 1      |
    /// | 2       | 1   | 2 | 3      |
    /// | 3       | 3   | 3 | 6      |
    ///
    /// And so, our final result, `6`.
    ///
    /// This example demonstrates the left-associative nature of `fold()`:
    /// it builds a string, starting with an initial value
    /// and continuing with each element from the front until the back:
    ///
    /// ```
    /// let numbers = [1, 2, 3, 4, 5];
    ///
    /// let zero = "0".to_string();
    ///
    /// let result = numbers.iter().fold(zero, |acc, &x| {
    ///     format!("({acc} + {x})")
    /// });
    ///
    /// assert_eq!(result, "(((((0 + 1) + 2) + 3) + 4) + 5)");
    /// ```
    /// It's common for people who haven't used iterators a lot to
    /// use a `for` loop with a list of things to build up a result. Those
    /// can be turned into `fold()`s:
    ///
    /// [`for`]: ../../book/ch03-05-control-flow.html#looping-through-a-collection-with-for
    ///
    /// ```
    /// let numbers = [1, 2, 3, 4, 5];
    ///
    /// let mut result = 0;
    ///
    /// // for loop:
    /// for i in &numbers {
    ///     result = result + i;
    /// }
    ///
    /// // fold:
    /// let result2 = numbers.iter().fold(0, |acc, &x| acc + x);
    ///
    /// // they're the same
    /// assert_eq!(result, result2);
    /// ```
    ///
    /// [`reduce()`]: Iterator::reduce
    #[doc(alias = "inject", alias = "foldl")]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }
        accum
    }

    /// Reduces the elements to a single one, by repeatedly applying a reducing
    /// operation.
    ///
    /// If the iterator is empty, returns [`None`]; otherwise, returns the
    /// result of the reduction.
    ///
    /// The reducing function is a closure with two arguments: an 'accumulator', and an element.
    /// For iterators with at least one element, this is the same as [`fold()`]
    /// with the first element of the iterator as the initial accumulator value, folding
    /// every subsequent element into it.
    ///
    /// [`fold()`]: Iterator::fold
    ///
    /// # Example
    ///
    /// ```
    /// let reduced: i32 = (1..10).reduce(|acc, e| acc + e).unwrap_or(0);
    /// assert_eq!(reduced, 45);
    ///
    /// // Which is equivalent to doing it with `fold`:
    /// let folded: i32 = (1..10).fold(0, |acc, e| acc + e);
    /// assert_eq!(reduced, folded);
    /// ```
    #[inline]
    #[stable(feature = "iterator_fold_self", since = "1.51.0")]
    fn reduce<F>(mut self, f: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        let first = self.next()?;
        Some(self.fold(first, f))
    }

    /// Reduces the elements to a single one by repeatedly applying a reducing operation. If the
    /// closure returns a failure, the failure is propagated back to the caller immediately.
    ///
    /// The return type of this method depends on the return type of the closure. If the closure
    /// returns `Result<Self::Item, E>`, then this function will return `Result<Option<Self::Item>,
    /// E>`. If the closure returns `Option<Self::Item>`, then this function will return
    /// `Option<Option<Self::Item>>`.
    ///
    /// When called on an empty iterator, this function will return either `Some(None)` or
    /// `Ok(None)` depending on the type of the provided closure.
    ///
    /// For iterators with at least one element, this is essentially the same as calling
    /// [`try_fold()`] with the first element of the iterator as the initial accumulator value.
    ///
    /// [`try_fold()`]: Iterator::try_fold
    ///
    /// # Examples
    ///
    /// Safely calculate the sum of a series of numbers:
    ///
    /// ```
    /// #![feature(iterator_try_reduce)]
    ///
    /// let numbers: Vec<usize> = vec![10, 20, 5, 23, 0];
    /// let sum = numbers.into_iter().try_reduce(|x, y| x.checked_add(y));
    /// assert_eq!(sum, Some(Some(58)));
    /// ```
    ///
    /// Determine when a reduction short circuited:
    ///
    /// ```
    /// #![feature(iterator_try_reduce)]
    ///
    /// let numbers = vec![1, 2, 3, usize::MAX, 4, 5];
    /// let sum = numbers.into_iter().try_reduce(|x, y| x.checked_add(y));
    /// assert_eq!(sum, None);
    /// ```
    ///
    /// Determine when a reduction was not performed because there are no elements:
    ///
    /// ```
    /// #![feature(iterator_try_reduce)]
    ///
    /// let numbers: Vec<usize> = Vec::new();
    /// let sum = numbers.into_iter().try_reduce(|x, y| x.checked_add(y));
    /// assert_eq!(sum, Some(None));
    /// ```
    ///
    /// Use a [`Result`] instead of an [`Option`]:
    ///
    /// ```
    /// #![feature(iterator_try_reduce)]
    ///
    /// let numbers = vec!["1", "2", "3", "4", "5"];
    /// let max: Result<Option<_>, <usize as std::str::FromStr>::Err> =
    ///     numbers.into_iter().try_reduce(|x, y| {
    ///         if x.parse::<usize>()? > y.parse::<usize>()? { Ok(x) } else { Ok(y) }
    ///     });
    /// assert_eq!(max, Ok(Some("5")));
    /// ```
    #[inline]
    #[unstable(feature = "iterator_try_reduce", reason = "new API", issue = "87053")]
    fn try_reduce<R>(
        &mut self,
        f: impl FnMut(Self::Item, Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<R::Output>>
    where
        Self: Sized,
        R: Try<Output = Self::Item, Residual: Residual<Option<Self::Item>>>,
    {
        let first = match self.next() {
            Some(i) => i,
            None => return Try::from_output(None),
        };

        match self.try_fold(first, f).branch() {
            ControlFlow::Break(r) => FromResidual::from_residual(r),
            ControlFlow::Continue(i) => Try::from_output(Some(i)),
        }
    }

    /// Tests if every element of the iterator matches a predicate.
    ///
    /// `all()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, and if they all return
    /// `true`, then so does `all()`. If any of them return `false`, it
    /// returns `false`.
    ///
    /// `all()` is short-circuiting; in other words, it will stop processing
    /// as soon as it finds a `false`, given that no matter what else happens,
    /// the result will also be `false`.
    ///
    /// An empty iterator returns `true`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert!(a.into_iter().all(|x| x > 0));
    ///
    /// assert!(!a.into_iter().all(|x| x > 2));
    /// ```
    ///
    /// Stopping at the first `false`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// assert!(!iter.all(|x| x != 2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn all<F>(&mut self, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        #[inline]
        fn check<T>(mut f: impl FnMut(T) -> bool) -> impl FnMut((), T) -> ControlFlow<()> {
            move |(), x| {
                if f(x) { ControlFlow::Continue(()) } else { ControlFlow::Break(()) }
            }
        }
        self.try_fold((), check(f)) == ControlFlow::Continue(())
    }

    /// Tests if any element of the iterator matches a predicate.
    ///
    /// `any()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, and if any of them return
    /// `true`, then so does `any()`. If they all return `false`, it
    /// returns `false`.
    ///
    /// `any()` is short-circuiting; in other words, it will stop processing
    /// as soon as it finds a `true`, given that no matter what else happens,
    /// the result will also be `true`.
    ///
    /// An empty iterator returns `false`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert!(a.into_iter().any(|x| x > 0));
    ///
    /// assert!(!a.into_iter().any(|x| x > 5));
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// assert!(iter.any(|x| x != 2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(2));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn any<F>(&mut self, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> bool,
    {
        #[inline]
        fn check<T>(mut f: impl FnMut(T) -> bool) -> impl FnMut((), T) -> ControlFlow<()> {
            move |(), x| {
                if f(x) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
            }
        }

        self.try_fold((), check(f)) == ControlFlow::Break(())
    }

    /// Searches for an element of an iterator that satisfies a predicate.
    ///
    /// `find()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, and if any of them return
    /// `true`, then `find()` returns [`Some(element)`]. If they all return
    /// `false`, it returns [`None`].
    ///
    /// `find()` is short-circuiting; in other words, it will stop processing
    /// as soon as the closure returns `true`.
    ///
    /// Because `find()` takes a reference, and many iterators iterate over
    /// references, this leads to a possibly confusing situation where the
    /// argument is a double reference. You can see this effect in the
    /// examples below, with `&&x`.
    ///
    /// If you need the index of the element, see [`position()`].
    ///
    /// [`Some(element)`]: Some
    /// [`position()`]: Iterator::position
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.into_iter().find(|&x| x == 2), Some(2));
    /// assert_eq!(a.into_iter().find(|&x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// assert_eq!(iter.find(|&x| x == 2), Some(2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(3));
    /// ```
    ///
    /// Note that `iter.find(f)` is equivalent to `iter.filter(f).next()`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        #[inline]
        fn check<T>(mut predicate: impl FnMut(&T) -> bool) -> impl FnMut((), T) -> ControlFlow<T> {
            move |(), x| {
                if predicate(&x) { ControlFlow::Break(x) } else { ControlFlow::Continue(()) }
            }
        }

        self.try_fold((), check(predicate)).break_value()
    }

    /// Applies function to the elements of iterator and returns
    /// the first non-none result.
    ///
    /// `iter.find_map(f)` is equivalent to `iter.filter_map(f).next()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = ["lol", "NaN", "2", "5"];
    ///
    /// let first_number = a.iter().find_map(|s| s.parse().ok());
    ///
    /// assert_eq!(first_number, Some(2));
    /// ```
    #[inline]
    #[stable(feature = "iterator_find_map", since = "1.30.0")]
    fn find_map<B, F>(&mut self, f: F) -> Option<B>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<B>,
    {
        #[inline]
        fn check<T, B>(mut f: impl FnMut(T) -> Option<B>) -> impl FnMut((), T) -> ControlFlow<B> {
            move |(), x| match f(x) {
                Some(x) => ControlFlow::Break(x),
                None => ControlFlow::Continue(()),
            }
        }

        self.try_fold((), check(f)).break_value()
    }

    /// Applies function to the elements of iterator and returns
    /// the first true result or the first error.
    ///
    /// The return type of this method depends on the return type of the closure.
    /// If you return `Result<bool, E>` from the closure, you'll get a `Result<Option<Self::Item>, E>`.
    /// If you return `Option<bool>` from the closure, you'll get an `Option<Option<Self::Item>>`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_find)]
    ///
    /// let a = ["1", "2", "lol", "NaN", "5"];
    ///
    /// let is_my_num = |s: &str, search: i32| -> Result<bool, std::num::ParseIntError> {
    ///     Ok(s.parse::<i32>()? == search)
    /// };
    ///
    /// let result = a.into_iter().try_find(|&s| is_my_num(s, 2));
    /// assert_eq!(result, Ok(Some("2")));
    ///
    /// let result = a.into_iter().try_find(|&s| is_my_num(s, 5));
    /// assert!(result.is_err());
    /// ```
    ///
    /// This also supports other types which implement [`Try`], not just [`Result`].
    ///
    /// ```
    /// #![feature(try_find)]
    ///
    /// use std::num::NonZero;
    ///
    /// let a = [3, 5, 7, 4, 9, 0, 11u32];
    /// let result = a.into_iter().try_find(|&x| NonZero::new(x).map(|y| y.is_power_of_two()));
    /// assert_eq!(result, Some(Some(4)));
    /// let result = a.into_iter().take(3).try_find(|&x| NonZero::new(x).map(|y| y.is_power_of_two()));
    /// assert_eq!(result, Some(None));
    /// let result = a.into_iter().rev().try_find(|&x| NonZero::new(x).map(|y| y.is_power_of_two()));
    /// assert_eq!(result, None);
    /// ```
    #[inline]
    #[unstable(feature = "try_find", reason = "new API", issue = "63178")]
    fn try_find<R>(
        &mut self,
        f: impl FnMut(&Self::Item) -> R,
    ) -> ChangeOutputType<R, Option<Self::Item>>
    where
        Self: Sized,
        R: Try<Output = bool, Residual: Residual<Option<Self::Item>>>,
    {
        #[inline]
        fn check<I, V, R>(
            mut f: impl FnMut(&I) -> V,
        ) -> impl FnMut((), I) -> ControlFlow<R::TryType>
        where
            V: Try<Output = bool, Residual = R>,
            R: Residual<Option<I>>,
        {
            move |(), x| match f(&x).branch() {
                ControlFlow::Continue(false) => ControlFlow::Continue(()),
                ControlFlow::Continue(true) => ControlFlow::Break(Try::from_output(Some(x))),
                ControlFlow::Break(r) => ControlFlow::Break(FromResidual::from_residual(r)),
            }
        }

        match self.try_fold((), check(f)) {
            ControlFlow::Break(x) => x,
            ControlFlow::Continue(()) => Try::from_output(None),
        }
    }

    /// Searches for an element in an iterator, returning its index.
    ///
    /// `position()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, and if one of them
    /// returns `true`, then `position()` returns [`Some(index)`]. If all of
    /// them return `false`, it returns [`None`].
    ///
    /// `position()` is short-circuiting; in other words, it will stop
    /// processing as soon as it finds a `true`.
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so if there are more
    /// than [`usize::MAX`] non-matching elements, it either produces the wrong
    /// result or panics. If overflow checks are enabled, a panic is
    /// guaranteed.
    ///
    /// # Panics
    ///
    /// This function might panic if the iterator has more than `usize::MAX`
    /// non-matching elements.
    ///
    /// [`Some(index)`]: Some
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.into_iter().position(|x| x == 2), Some(1));
    ///
    /// assert_eq!(a.into_iter().position(|x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3, 4];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// assert_eq!(iter.position(|x| x >= 2), Some(1));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(3));
    ///
    /// // The returned index depends on iterator state
    /// assert_eq!(iter.position(|x| x == 4), Some(0));
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        #[inline]
        fn check<'a, T>(
            mut predicate: impl FnMut(T) -> bool + 'a,
            acc: &'a mut usize,
        ) -> impl FnMut((), T) -> ControlFlow<usize, ()> + 'a {
            #[rustc_inherit_overflow_checks]
            move |_, x| {
                if predicate(x) {
                    ControlFlow::Break(*acc)
                } else {
                    *acc += 1;
                    ControlFlow::Continue(())
                }
            }
        }

        let mut acc = 0;
        self.try_fold((), check(predicate, &mut acc)).break_value()
    }

    /// Searches for an element in an iterator from the right, returning its
    /// index.
    ///
    /// `rposition()` takes a closure that returns `true` or `false`. It applies
    /// this closure to each element of the iterator, starting from the end,
    /// and if one of them returns `true`, then `rposition()` returns
    /// [`Some(index)`]. If all of them return `false`, it returns [`None`].
    ///
    /// `rposition()` is short-circuiting; in other words, it will stop
    /// processing as soon as it finds a `true`.
    ///
    /// [`Some(index)`]: Some
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.into_iter().rposition(|x| x == 3), Some(2));
    ///
    /// assert_eq!(a.into_iter().rposition(|x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [-1, 2, 3, 4];
    ///
    /// let mut iter = a.into_iter();
    ///
    /// assert_eq!(iter.rposition(|x| x >= 2), Some(3));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(-1));
    /// assert_eq!(iter.next_back(), Some(3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rposition<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
        Self: Sized + ExactSizeIterator + DoubleEndedIterator,
    {
        // No need for an overflow check here, because `ExactSizeIterator`
        // implies that the number of elements fits into a `usize`.
        #[inline]
        fn check<T>(
            mut predicate: impl FnMut(T) -> bool,
        ) -> impl FnMut(usize, T) -> ControlFlow<usize, usize> {
            move |i, x| {
                let i = i - 1;
                if predicate(x) { ControlFlow::Break(i) } else { ControlFlow::Continue(i) }
            }
        }

        let n = self.len();
        self.try_rfold(n, check(predicate)).break_value()
    }

    /// Returns the maximum element of an iterator.
    ///
    /// If several elements are equally maximum, the last element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// Note that [`f32`]/[`f64`] doesn't implement [`Ord`] due to NaN being
    /// incomparable. You can work around this by using [`Iterator::reduce`]:
    /// ```
    /// assert_eq!(
    ///     [2.4, f32::NAN, 1.3]
    ///         .into_iter()
    ///         .reduce(f32::max)
    ///         .unwrap_or(0.),
    ///     2.4
    /// );
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let b: [u32; 0] = [];
    ///
    /// assert_eq!(a.into_iter().max(), Some(3));
    /// assert_eq!(b.into_iter().max(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        self.max_by(Ord::cmp)
    }

    /// Returns the minimum element of an iterator.
    ///
    /// If several elements are equally minimum, the first element is returned.
    /// If the iterator is empty, [`None`] is returned.
    ///
    /// Note that [`f32`]/[`f64`] doesn't implement [`Ord`] due to NaN being
    /// incomparable. You can work around this by using [`Iterator::reduce`]:
    /// ```
    /// assert_eq!(
    ///     [2.4, f32::NAN, 1.3]
    ///         .into_iter()
    ///         .reduce(f32::min)
    ///         .unwrap_or(0.),
    ///     1.3
    /// );
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let b: [u32; 0] = [];
    ///
    /// assert_eq!(a.into_iter().min(), Some(1));
    /// assert_eq!(b.into_iter().min(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        self.min_by(Ord::cmp)
    }

    /// Returns the element that gives the maximum value from the
    /// specified function.
    ///
    /// If several elements are equally maximum, the last element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(a.into_iter().max_by_key(|x| x.abs()).unwrap(), -10);
    /// ```
    #[inline]
    #[stable(feature = "iter_cmp_by_key", since = "1.6.0")]
    fn max_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        #[inline]
        fn key<T, B>(mut f: impl FnMut(&T) -> B) -> impl FnMut(T) -> (B, T) {
            move |x| (f(&x), x)
        }

        #[inline]
        fn compare<T, B: Ord>((x_p, _): &(B, T), (y_p, _): &(B, T)) -> Ordering {
            x_p.cmp(y_p)
        }

        let (_, x) = self.map(key(f)).max_by(compare)?;
        Some(x)
    }

    /// Returns the element that gives the maximum value with respect to the
    /// specified comparison function.
    ///
    /// If several elements are equally maximum, the last element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(a.into_iter().max_by(|x, y| x.cmp(y)).unwrap(), 5);
    /// ```
    #[inline]
    #[stable(feature = "iter_max_by", since = "1.15.0")]
    fn max_by<F>(self, compare: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        #[inline]
        fn fold<T>(mut compare: impl FnMut(&T, &T) -> Ordering) -> impl FnMut(T, T) -> T {
            move |x, y| cmp::max_by(x, y, &mut compare)
        }

        self.reduce(fold(compare))
    }

    /// Returns the element that gives the minimum value from the
    /// specified function.
    ///
    /// If several elements are equally minimum, the first element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(a.into_iter().min_by_key(|x| x.abs()).unwrap(), 0);
    /// ```
    #[inline]
    #[stable(feature = "iter_cmp_by_key", since = "1.6.0")]
    fn min_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        #[inline]
        fn key<T, B>(mut f: impl FnMut(&T) -> B) -> impl FnMut(T) -> (B, T) {
            move |x| (f(&x), x)
        }

        #[inline]
        fn compare<T, B: Ord>((x_p, _): &(B, T), (y_p, _): &(B, T)) -> Ordering {
            x_p.cmp(y_p)
        }

        let (_, x) = self.map(key(f)).min_by(compare)?;
        Some(x)
    }

    /// Returns the element that gives the minimum value with respect to the
    /// specified comparison function.
    ///
    /// If several elements are equally minimum, the first element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(a.into_iter().min_by(|x, y| x.cmp(y)).unwrap(), -10);
    /// ```
    #[inline]
    #[stable(feature = "iter_min_by", since = "1.15.0")]
    fn min_by<F>(self, compare: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        #[inline]
        fn fold<T>(mut compare: impl FnMut(&T, &T) -> Ordering) -> impl FnMut(T, T) -> T {
            move |x, y| cmp::min_by(x, y, &mut compare)
        }

        self.reduce(fold(compare))
    }

    /// Reverses an iterator's direction.
    ///
    /// Usually, iterators iterate from left to right. After using `rev()`,
    /// an iterator will instead iterate from right to left.
    ///
    /// This is only possible if the iterator has an end, so `rev()` only
    /// works on [`DoubleEndedIterator`]s.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter().rev();
    ///
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(1));
    ///
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[doc(alias = "reverse")]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rev(self) -> Rev<Self>
    where
        Self: Sized + DoubleEndedIterator,
    {
        Rev::new(self)
    }

    /// Converts an iterator of pairs into a pair of containers.
    ///
    /// `unzip()` consumes an entire iterator of pairs, producing two
    /// collections: one from the left elements of the pairs, and one
    /// from the right elements.
    ///
    /// This function is, in some sense, the opposite of [`zip`].
    ///
    /// [`zip`]: Iterator::zip
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [(1, 2), (3, 4), (5, 6)];
    ///
    /// let (left, right): (Vec<_>, Vec<_>) = a.into_iter().unzip();
    ///
    /// assert_eq!(left, [1, 3, 5]);
    /// assert_eq!(right, [2, 4, 6]);
    ///
    /// // you can also unzip multiple nested tuples at once
    /// let a = [(1, (2, 3)), (4, (5, 6))];
    ///
    /// let (x, (y, z)): (Vec<_>, (Vec<_>, Vec<_>)) = a.into_iter().unzip();
    /// assert_eq!(x, [1, 4]);
    /// assert_eq!(y, [2, 5]);
    /// assert_eq!(z, [3, 6]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
    where
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
        Self: Sized + Iterator<Item = (A, B)>,
    {
        let mut unzipped: (FromA, FromB) = Default::default();
        unzipped.extend(self);
        unzipped
    }

    /// Creates an iterator which copies all of its elements.
    ///
    /// This is useful when you have an iterator over `&T`, but you need an
    /// iterator over `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let v_copied: Vec<_> = a.iter().copied().collect();
    ///
    /// // copied is the same as .map(|&x| x)
    /// let v_map: Vec<_> = a.iter().map(|&x| x).collect();
    ///
    /// assert_eq!(v_copied, [1, 2, 3]);
    /// assert_eq!(v_map, [1, 2, 3]);
    /// ```
    #[stable(feature = "iter_copied", since = "1.36.0")]
    #[rustc_diagnostic_item = "iter_copied"]
    fn copied<'a, T: 'a>(self) -> Copied<Self>
    where
        Self: Sized + Iterator<Item = &'a T>,
        T: Copy,
    {
        Copied::new(self)
    }

    /// Creates an iterator which [`clone`]s all of its elements.
    ///
    /// This is useful when you have an iterator over `&T`, but you need an
    /// iterator over `T`.
    ///
    /// There is no guarantee whatsoever about the `clone` method actually
    /// being called *or* optimized away. So code should not depend on
    /// either.
    ///
    /// [`clone`]: Clone::clone
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let v_cloned: Vec<_> = a.iter().cloned().collect();
    ///
    /// // cloned is the same as .map(|&x| x), for integers
    /// let v_map: Vec<_> = a.iter().map(|&x| x).collect();
    ///
    /// assert_eq!(v_cloned, [1, 2, 3]);
    /// assert_eq!(v_map, [1, 2, 3]);
    /// ```
    ///
    /// To get the best performance, try to clone late:
    ///
    /// ```
    /// let a = [vec![0_u8, 1, 2], vec![3, 4], vec![23]];
    /// // don't do this:
    /// let slower: Vec<_> = a.iter().cloned().filter(|s| s.len() == 1).collect();
    /// assert_eq!(&[vec![23]], &slower[..]);
    /// // instead call `cloned` late
    /// let faster: Vec<_> = a.iter().filter(|s| s.len() == 1).cloned().collect();
    /// assert_eq!(&[vec![23]], &faster[..]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "iter_cloned"]
    fn cloned<'a, T: 'a>(self) -> Cloned<Self>
    where
        Self: Sized + Iterator<Item = &'a T>,
        T: Clone,
    {
        Cloned::new(self)
    }

    /// Repeats an iterator endlessly.
    ///
    /// Instead of stopping at [`None`], the iterator will instead start again,
    /// from the beginning. After iterating again, it will start at the
    /// beginning again. And again. And again. Forever. Note that in case the
    /// original iterator is empty, the resulting iterator will also be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter().cycle();
    ///
    /// loop {
    ///     assert_eq!(iter.next(), Some(1));
    ///     assert_eq!(iter.next(), Some(2));
    ///     assert_eq!(iter.next(), Some(3));
    /// #   break;
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn cycle(self) -> Cycle<Self>
    where
        Self: Sized + Clone,
    {
        Cycle::new(self)
    }

    /// Returns an iterator over `N` elements of the iterator at a time.
    ///
    /// The chunks do not overlap. If `N` does not divide the length of the
    /// iterator, then the last up to `N-1` elements will be omitted and can be
    /// retrieved from the [`.into_remainder()`][ArrayChunks::into_remainder]
    /// function of the iterator.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_array_chunks)]
    ///
    /// let mut iter = "lorem".chars().array_chunks();
    /// assert_eq!(iter.next(), Some(['l', 'o']));
    /// assert_eq!(iter.next(), Some(['r', 'e']));
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.into_remainder().unwrap().as_slice(), &['m']);
    /// ```
    ///
    /// ```
    /// #![feature(iter_array_chunks)]
    ///
    /// let data = [1, 1, 2, -2, 6, 0, 3, 1];
    /// //          ^-----^  ^------^
    /// for [x, y, z] in data.iter().array_chunks() {
    ///     assert_eq!(x + y + z, 4);
    /// }
    /// ```
    #[track_caller]
    #[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
    fn array_chunks<const N: usize>(self) -> ArrayChunks<Self, N>
    where
        Self: Sized,
    {
        ArrayChunks::new(self)
    }

    /// Sums the elements of an iterator.
    ///
    /// Takes each element, adds them together, and returns the result.
    ///
    /// An empty iterator returns the *additive identity* ("zero") of the type,
    /// which is `0` for integers and `-0.0` for floats.
    ///
    /// `sum()` can be used to sum any type implementing [`Sum`][`core::iter::Sum`],
    /// including [`Option`][`Option::sum`] and [`Result`][`Result::sum`].
    ///
    /// # Panics
    ///
    /// When calling `sum()` and a primitive integer type is being returned, this
    /// method will panic if the computation overflows and overflow checks are
    /// enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let sum: i32 = a.iter().sum();
    ///
    /// assert_eq!(sum, 6);
    ///
    /// let b: Vec<f32> = vec![];
    /// let sum: f32 = b.iter().sum();
    /// assert_eq!(sum, -0.0_f32);
    /// ```
    #[stable(feature = "iter_arith", since = "1.11.0")]
    fn sum<S>(self) -> S
    where
        Self: Sized,
        S: Sum<Self::Item>,
    {
        Sum::sum(self)
    }

    /// Iterates over the entire iterator, multiplying all the elements
    ///
    /// An empty iterator returns the one value of the type.
    ///
    /// `product()` can be used to multiply any type implementing [`Product`][`core::iter::Product`],
    /// including [`Option`][`Option::product`] and [`Result`][`Result::product`].
    ///
    /// # Panics
    ///
    /// When calling `product()` and a primitive integer type is being returned,
    /// method will panic if the computation overflows and overflow checks are
    /// enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// fn factorial(n: u32) -> u32 {
    ///     (1..=n).product()
    /// }
    /// assert_eq!(factorial(0), 1);
    /// assert_eq!(factorial(1), 1);
    /// assert_eq!(factorial(5), 120);
    /// ```
    #[stable(feature = "iter_arith", since = "1.11.0")]
    fn product<P>(self) -> P
    where
        Self: Sized,
        P: Product<Self::Item>,
    {
        Product::product(self)
    }

    /// [Lexicographically](Ord#lexicographical-comparison) compares the elements of this [`Iterator`] with those
    /// of another.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!([1].iter().cmp([1].iter()), Ordering::Equal);
    /// assert_eq!([1].iter().cmp([1, 2].iter()), Ordering::Less);
    /// assert_eq!([1, 2].iter().cmp([1].iter()), Ordering::Greater);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn cmp<I>(self, other: I) -> Ordering
    where
        I: IntoIterator<Item = Self::Item>,
        Self::Item: Ord,
        Self: Sized,
    {
        self.cmp_by(other, |x, y| x.cmp(&y))
    }

    /// [Lexicographically](Ord#lexicographical-comparison) compares the elements of this [`Iterator`] with those
    /// of another with respect to the specified comparison function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_order_by)]
    ///
    /// use std::cmp::Ordering;
    ///
    /// let xs = [1, 2, 3, 4];
    /// let ys = [1, 4, 9, 16];
    ///
    /// assert_eq!(xs.into_iter().cmp_by(ys, |x, y| x.cmp(&y)), Ordering::Less);
    /// assert_eq!(xs.into_iter().cmp_by(ys, |x, y| (x * x).cmp(&y)), Ordering::Equal);
    /// assert_eq!(xs.into_iter().cmp_by(ys, |x, y| (2 * x).cmp(&y)), Ordering::Greater);
    /// ```
    #[unstable(feature = "iter_order_by", issue = "64295")]
    fn cmp_by<I, F>(self, other: I, cmp: F) -> Ordering
    where
        Self: Sized,
        I: IntoIterator,
        F: FnMut(Self::Item, I::Item) -> Ordering,
    {
        #[inline]
        fn compare<X, Y, F>(mut cmp: F) -> impl FnMut(X, Y) -> ControlFlow<Ordering>
        where
            F: FnMut(X, Y) -> Ordering,
        {
            move |x, y| match cmp(x, y) {
                Ordering::Equal => ControlFlow::Continue(()),
                non_eq => ControlFlow::Break(non_eq),
            }
        }

        match iter_compare(self, other.into_iter(), compare(cmp)) {
            ControlFlow::Continue(ord) => ord,
            ControlFlow::Break(ord) => ord,
        }
    }

    /// [Lexicographically](Ord#lexicographical-comparison) compares the [`PartialOrd`] elements of
    /// this [`Iterator`] with those of another. The comparison works like short-circuit
    /// evaluation, returning a result without comparing the remaining elements.
    /// As soon as an order can be determined, the evaluation stops and a result is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!([1.].iter().partial_cmp([1.].iter()), Some(Ordering::Equal));
    /// assert_eq!([1.].iter().partial_cmp([1., 2.].iter()), Some(Ordering::Less));
    /// assert_eq!([1., 2.].iter().partial_cmp([1.].iter()), Some(Ordering::Greater));
    /// ```
    ///
    /// For floating-point numbers, NaN does not have a total order and will result
    /// in `None` when compared:
    ///
    /// ```
    /// assert_eq!([f64::NAN].iter().partial_cmp([1.].iter()), None);
    /// ```
    ///
    /// The results are determined by the order of evaluation.
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!([1.0, f64::NAN].iter().partial_cmp([2.0, f64::NAN].iter()), Some(Ordering::Less));
    /// assert_eq!([2.0, f64::NAN].iter().partial_cmp([1.0, f64::NAN].iter()), Some(Ordering::Greater));
    /// assert_eq!([f64::NAN, 1.0].iter().partial_cmp([f64::NAN, 2.0].iter()), None);
    /// ```
    ///
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn partial_cmp<I>(self, other: I) -> Option<Ordering>
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        self.partial_cmp_by(other, |x, y| x.partial_cmp(&y))
    }

    /// [Lexicographically](Ord#lexicographical-comparison) compares the elements of this [`Iterator`] with those
    /// of another with respect to the specified comparison function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_order_by)]
    ///
    /// use std::cmp::Ordering;
    ///
    /// let xs = [1.0, 2.0, 3.0, 4.0];
    /// let ys = [1.0, 4.0, 9.0, 16.0];
    ///
    /// assert_eq!(
    ///     xs.iter().partial_cmp_by(ys, |x, y| x.partial_cmp(&y)),
    ///     Some(Ordering::Less)
    /// );
    /// assert_eq!(
    ///     xs.iter().partial_cmp_by(ys, |x, y| (x * x).partial_cmp(&y)),
    ///     Some(Ordering::Equal)
    /// );
    /// assert_eq!(
    ///     xs.iter().partial_cmp_by(ys, |x, y| (2.0 * x).partial_cmp(&y)),
    ///     Some(Ordering::Greater)
    /// );
    /// ```
    #[unstable(feature = "iter_order_by", issue = "64295")]
    fn partial_cmp_by<I, F>(self, other: I, partial_cmp: F) -> Option<Ordering>
    where
        Self: Sized,
        I: IntoIterator,
        F: FnMut(Self::Item, I::Item) -> Option<Ordering>,
    {
        #[inline]
        fn compare<X, Y, F>(mut partial_cmp: F) -> impl FnMut(X, Y) -> ControlFlow<Option<Ordering>>
        where
            F: FnMut(X, Y) -> Option<Ordering>,
        {
            move |x, y| match partial_cmp(x, y) {
                Some(Ordering::Equal) => ControlFlow::Continue(()),
                non_eq => ControlFlow::Break(non_eq),
            }
        }

        match iter_compare(self, other.into_iter(), compare(partial_cmp)) {
            ControlFlow::Continue(ord) => Some(ord),
            ControlFlow::Break(ord) => ord,
        }
    }

    /// Determines if the elements of this [`Iterator`] are equal to those of
    /// another.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([1].iter().eq([1].iter()), true);
    /// assert_eq!([1].iter().eq([1, 2].iter()), false);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn eq<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
        Self: Sized,
    {
        self.eq_by(other, |x, y| x == y)
    }

    /// Determines if the elements of this [`Iterator`] are equal to those of
    /// another with respect to the specified equality function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_order_by)]
    ///
    /// let xs = [1, 2, 3, 4];
    /// let ys = [1, 4, 9, 16];
    ///
    /// assert!(xs.iter().eq_by(ys, |x, y| x * x == y));
    /// ```
    #[unstable(feature = "iter_order_by", issue = "64295")]
    fn eq_by<I, F>(self, other: I, eq: F) -> bool
    where
        Self: Sized,
        I: IntoIterator,
        F: FnMut(Self::Item, I::Item) -> bool,
    {
        #[inline]
        fn compare<X, Y, F>(mut eq: F) -> impl FnMut(X, Y) -> ControlFlow<()>
        where
            F: FnMut(X, Y) -> bool,
        {
            move |x, y| {
                if eq(x, y) { ControlFlow::Continue(()) } else { ControlFlow::Break(()) }
            }
        }

        match iter_compare(self, other.into_iter(), compare(eq)) {
            ControlFlow::Continue(ord) => ord == Ordering::Equal,
            ControlFlow::Break(()) => false,
        }
    }

    /// Determines if the elements of this [`Iterator`] are not equal to those of
    /// another.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([1].iter().ne([1].iter()), false);
    /// assert_eq!([1].iter().ne([1, 2].iter()), true);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn ne<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
        Self: Sized,
    {
        !self.eq(other)
    }

    /// Determines if the elements of this [`Iterator`] are [lexicographically](Ord#lexicographical-comparison)
    /// less than those of another.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([1].iter().lt([1].iter()), false);
    /// assert_eq!([1].iter().lt([1, 2].iter()), true);
    /// assert_eq!([1, 2].iter().lt([1].iter()), false);
    /// assert_eq!([1, 2].iter().lt([1, 2].iter()), false);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn lt<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        self.partial_cmp(other) == Some(Ordering::Less)
    }

    /// Determines if the elements of this [`Iterator`] are [lexicographically](Ord#lexicographical-comparison)
    /// less or equal to those of another.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([1].iter().le([1].iter()), true);
    /// assert_eq!([1].iter().le([1, 2].iter()), true);
    /// assert_eq!([1, 2].iter().le([1].iter()), false);
    /// assert_eq!([1, 2].iter().le([1, 2].iter()), true);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn le<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        matches!(self.partial_cmp(other), Some(Ordering::Less | Ordering::Equal))
    }

    /// Determines if the elements of this [`Iterator`] are [lexicographically](Ord#lexicographical-comparison)
    /// greater than those of another.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([1].iter().gt([1].iter()), false);
    /// assert_eq!([1].iter().gt([1, 2].iter()), false);
    /// assert_eq!([1, 2].iter().gt([1].iter()), true);
    /// assert_eq!([1, 2].iter().gt([1, 2].iter()), false);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn gt<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        self.partial_cmp(other) == Some(Ordering::Greater)
    }

    /// Determines if the elements of this [`Iterator`] are [lexicographically](Ord#lexicographical-comparison)
    /// greater than or equal to those of another.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!([1].iter().ge([1].iter()), true);
    /// assert_eq!([1].iter().ge([1, 2].iter()), false);
    /// assert_eq!([1, 2].iter().ge([1].iter()), true);
    /// assert_eq!([1, 2].iter().ge([1, 2].iter()), true);
    /// ```
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn ge<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        matches!(self.partial_cmp(other), Some(Ordering::Greater | Ordering::Equal))
    }

    /// Checks if the elements of this iterator are sorted.
    ///
    /// That is, for each element `a` and its following element `b`, `a <= b` must hold. If the
    /// iterator yields exactly zero or one element, `true` is returned.
    ///
    /// Note that if `Self::Item` is only `PartialOrd`, but not `Ord`, the above definition
    /// implies that this function returns `false` if any two consecutive items are not
    /// comparable.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!([1, 2, 2, 9].iter().is_sorted());
    /// assert!(![1, 3, 2, 4].iter().is_sorted());
    /// assert!([0].iter().is_sorted());
    /// assert!(std::iter::empty::<i32>().is_sorted());
    /// assert!(![0.0, 1.0, f32::NAN].iter().is_sorted());
    /// ```
    #[inline]
    #[stable(feature = "is_sorted", since = "1.82.0")]
    fn is_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        self.is_sorted_by(|a, b| a <= b)
    }

    /// Checks if the elements of this iterator are sorted using the given comparator function.
    ///
    /// Instead of using `PartialOrd::partial_cmp`, this function uses the given `compare`
    /// function to determine whether two elements are to be considered in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!([1, 2, 2, 9].iter().is_sorted_by(|a, b| a <= b));
    /// assert!(![1, 2, 2, 9].iter().is_sorted_by(|a, b| a < b));
    ///
    /// assert!([0].iter().is_sorted_by(|a, b| true));
    /// assert!([0].iter().is_sorted_by(|a, b| false));
    ///
    /// assert!(std::iter::empty::<i32>().is_sorted_by(|a, b| false));
    /// assert!(std::iter::empty::<i32>().is_sorted_by(|a, b| true));
    /// ```
    #[stable(feature = "is_sorted", since = "1.82.0")]
    fn is_sorted_by<F>(mut self, compare: F) -> bool
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> bool,
    {
        #[inline]
        fn check<'a, T>(
            last: &'a mut T,
            mut compare: impl FnMut(&T, &T) -> bool + 'a,
        ) -> impl FnMut(T) -> bool + 'a {
            move |curr| {
                if !compare(&last, &curr) {
                    return false;
                }
                *last = curr;
                true
            }
        }

        let mut last = match self.next() {
            Some(e) => e,
            None => return true,
        };

        self.all(check(&mut last, compare))
    }

    /// Checks if the elements of this iterator are sorted using the given key extraction
    /// function.
    ///
    /// Instead of comparing the iterator's elements directly, this function compares the keys of
    /// the elements, as determined by `f`. Apart from that, it's equivalent to [`is_sorted`]; see
    /// its documentation for more information.
    ///
    /// [`is_sorted`]: Iterator::is_sorted
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(["c", "bb", "aaa"].iter().is_sorted_by_key(|s| s.len()));
    /// assert!(![-2i32, -1, 0, 3].iter().is_sorted_by_key(|n| n.abs()));
    /// ```
    #[inline]
    #[stable(feature = "is_sorted", since = "1.82.0")]
    fn is_sorted_by_key<F, K>(self, f: F) -> bool
    where
        Self: Sized,
        F: FnMut(Self::Item) -> K,
        K: PartialOrd,
    {
        self.map(f).is_sorted()
    }

    /// See [TrustedRandomAccess][super::super::TrustedRandomAccess]
    // The unusual name is to avoid name collisions in method resolution
    // see #76479.
    #[inline]
    #[doc(hidden)]
    #[unstable(feature = "trusted_random_access", issue = "none")]
    unsafe fn __iterator_get_unchecked(&mut self, _idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        unreachable!("Always specialized");
    }
}

/// Compares two iterators element-wise using the given function.
///
/// If `ControlFlow::Continue(())` is returned from the function, the comparison moves on to the next
/// elements of both iterators. Returning `ControlFlow::Break(x)` short-circuits the iteration and
/// returns `ControlFlow::Break(x)`. If one of the iterators runs out of elements,
/// `ControlFlow::Continue(ord)` is returned where `ord` is the result of comparing the lengths of
/// the iterators.
///
/// Isolates the logic shared by ['cmp_by'](Iterator::cmp_by),
/// ['partial_cmp_by'](Iterator::partial_cmp_by), and ['eq_by'](Iterator::eq_by).
#[inline]
fn iter_compare<A, B, F, T>(mut a: A, mut b: B, f: F) -> ControlFlow<T, Ordering>
where
    A: Iterator,
    B: Iterator,
    F: FnMut(A::Item, B::Item) -> ControlFlow<T>,
{
    #[inline]
    fn compare<'a, B, X, T>(
        b: &'a mut B,
        mut f: impl FnMut(X, B::Item) -> ControlFlow<T> + 'a,
    ) -> impl FnMut(X) -> ControlFlow<ControlFlow<T, Ordering>> + 'a
    where
        B: Iterator,
    {
        move |x| match b.next() {
            None => ControlFlow::Break(ControlFlow::Continue(Ordering::Greater)),
            Some(y) => f(x, y).map_break(ControlFlow::Break),
        }
    }

    match a.try_for_each(compare(&mut b, f)) {
        ControlFlow::Continue(()) => ControlFlow::Continue(match b.next() {
            None => Ordering::Equal,
            Some(_) => Ordering::Less,
        }),
        ControlFlow::Break(x) => x,
    }
}

/// Implements `Iterator` for mutable references to iterators, such as those produced by [`Iterator::by_ref`].
///
/// This implementation passes all method calls on to the original iterator.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized> Iterator for &mut I {
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        (**self).next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        (**self).advance_by(n)
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth(n)
    }
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.spec_fold(init, f)
    }
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.spec_try_fold(init, f)
    }
}

/// Helper trait to specialize `fold` and `try_fold` for `&mut I where I: Sized`
trait IteratorRefSpec: Iterator {
    fn spec_fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B;

    fn spec_try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>;
}

impl<I: Iterator + ?Sized> IteratorRefSpec for &mut I {
    default fn spec_fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }
        accum
    }

    default fn spec_try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x)?;
        }
        try { accum }
    }
}

impl<I: Iterator> IteratorRefSpec for &mut I {
    impl_fold_via_try_fold! { spec_fold -> spec_try_fold }

    fn spec_try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        (**self).try_fold(init, f)
    }
}
