use crate::cmp::Ordering;
use crate::ops::Try;

use super::super::LoopState;
use super::super::{Chain, Cycle, Copied, Cloned, Enumerate, Filter, FilterMap, Fuse};
use super::super::{Flatten, FlatMap};
use super::super::{Inspect, Map, Peekable, Scan, Skip, SkipWhile, StepBy, Take, TakeWhile, Rev};
use super::super::{Zip, Sum, Product, FromIterator};

fn _assert_is_object_safe(_: &dyn Iterator<Item=()>) {}

/// An interface for dealing with iterators.
///
/// This is the main iterator trait. For more about the concept of iterators
/// generally, please see the [module-level documentation]. In particular, you
/// may want to know how to [implement `Iterator`][impl].
///
/// [module-level documentation]: index.html
/// [impl]: index.html#implementing-iterator
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented(
    on(
        _Self="[std::ops::Range<Idx>; 1]",
        label="if you meant to iterate between two values, remove the square brackets",
        note="`[start..end]` is an array of one `Range`; you might have meant to have a `Range` \
              without the brackets: `start..end`"
    ),
    on(
        _Self="[std::ops::RangeFrom<Idx>; 1]",
        label="if you meant to iterate from a value onwards, remove the square brackets",
        note="`[start..]` is an array of one `RangeFrom`; you might have meant to have a \
              `RangeFrom` without the brackets: `start..`, keeping in mind that iterating over an \
              unbounded iterator will run forever unless you `break` or `return` from within the \
              loop"
    ),
    on(
        _Self="[std::ops::RangeTo<Idx>; 1]",
        label="if you meant to iterate until a value, remove the square brackets and add a \
               starting value",
        note="`[..end]` is an array of one `RangeTo`; you might have meant to have a bounded \
              `Range` without the brackets: `0..end`"
    ),
    on(
        _Self="[std::ops::RangeInclusive<Idx>; 1]",
        label="if you meant to iterate between two values, remove the square brackets",
        note="`[start..=end]` is an array of one `RangeInclusive`; you might have meant to have a \
              `RangeInclusive` without the brackets: `start..=end`"
    ),
    on(
        _Self="[std::ops::RangeToInclusive<Idx>; 1]",
        label="if you meant to iterate until a value (including it), remove the square brackets \
               and add a starting value",
        note="`[..=end]` is an array of one `RangeToInclusive`; you might have meant to have a \
              bounded `RangeInclusive` without the brackets: `0..=end`"
    ),
    on(
        _Self="std::ops::RangeTo<Idx>",
        label="if you meant to iterate until a value, add a starting value",
        note="`..end` is a `RangeTo`, which cannot be iterated on; you might have meant to have a \
              bounded `Range`: `0..end`"
    ),
    on(
        _Self="std::ops::RangeToInclusive<Idx>",
        label="if you meant to iterate until a value (including it), add a starting value",
        note="`..=end` is a `RangeToInclusive`, which cannot be iterated on; you might have meant \
              to have a bounded `RangeInclusive`: `0..=end`"
    ),
    on(
        _Self="&str",
        label="`{Self}` is not an iterator; try calling `.chars()` or `.bytes()`"
    ),
    on(
        _Self="std::string::String",
        label="`{Self}` is not an iterator; try calling `.chars()` or `.bytes()`"
    ),
    on(
        _Self="[]",
        label="borrow the array with `&` or call `.iter()` on it to iterate over it",
        note="arrays are not iterators, but slices like the following are: `&[1, 2, 3]`"
    ),
    on(
        _Self="{integral}",
        note="if you want to iterate between `start` until a value `end`, use the exclusive range \
              syntax `start..end` or the inclusive range syntax `start..=end`"
    ),
    label="`{Self}` is not an iterator",
    message="`{Self}` is not an iterator"
)]
#[doc(spotlight)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub trait Iterator {
    /// The type of the elements being iterated over.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Advances the iterator and returns the next value.
    ///
    /// Returns [`None`] when iteration is finished. Individual iterator
    /// implementations may choose to resume iteration, and so calling `next()`
    /// again may or may not eventually start returning [`Some(Item)`] again at some
    /// point.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`Some(Item)`]: ../../std/option/enum.Option.html#variant.Some
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// // A call to next() returns the next value...
    /// assert_eq!(Some(&1), iter.next());
    /// assert_eq!(Some(&2), iter.next());
    /// assert_eq!(Some(&3), iter.next());
    ///
    /// // ... and then None once it's over.
    /// assert_eq!(None, iter.next());
    ///
    /// // More calls may or may not return `None`. Here, they always will.
    /// assert_eq!(None, iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next(&mut self) -> Option<Self::Item>;

    /// Returns the bounds on the remaining length of the iterator.
    ///
    /// Specifically, `size_hint()` returns a tuple where the first element
    /// is the lower bound, and the second element is the upper bound.
    ///
    /// The second half of the tuple that is returned is an [`Option`]`<`[`usize`]`>`.
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
    /// The default implementation returns `(0, `[`None`]`)` which is correct for any
    /// iterator.
    ///
    /// [`usize`]: ../../std/primitive.usize.html
    /// [`Option`]: ../../std/option/enum.Option.html
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let iter = a.iter();
    ///
    /// assert_eq!((3, Some(3)), iter.size_hint());
    /// ```
    ///
    /// A more complex example:
    ///
    /// ```
    /// // The even numbers from zero to ten.
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
    /// assert_eq!((usize::max_value(), None), iter.size_hint());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn size_hint(&self) -> (usize, Option<usize>) { (0, None) }

    /// Consumes the iterator, counting the number of iterations and returning it.
    ///
    /// This method will evaluate the iterator until its [`next`] returns
    /// [`None`]. Once [`None`] is encountered, `count()` returns the number of
    /// times it called [`next`].
    ///
    /// [`next`]: #tymethod.next
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so counting elements of
    /// an iterator with more than [`usize::MAX`] elements either produces the
    /// wrong result or panics. If debug assertions are enabled, a panic is
    /// guaranteed.
    ///
    /// # Panics
    ///
    /// This function might panic if the iterator has more than [`usize::MAX`]
    /// elements.
    ///
    /// [`usize::MAX`]: ../../std/usize/constant.MAX.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().count(), 3);
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().count(), 5);
    /// ```
    #[inline]
    #[rustc_inherit_overflow_checks]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn count(self) -> usize where Self: Sized {
        // Might overflow.
        self.fold(0, |cnt, _| cnt + 1)
    }

    /// Consumes the iterator, returning the last element.
    ///
    /// This method will evaluate the iterator until it returns [`None`]. While
    /// doing so, it keeps track of the current element. After [`None`] is
    /// returned, `last()` will then return the last element it saw.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().last(), Some(&3));
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().last(), Some(&5));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn last(self) -> Option<Self::Item> where Self: Sized {
        let mut last = None;
        for x in self { last = Some(x); }
        last
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
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().nth(1), Some(&2));
    /// ```
    ///
    /// Calling `nth()` multiple times doesn't rewind the iterator:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.nth(1), Some(&2));
    /// assert_eq!(iter.nth(1), None);
    /// ```
    ///
    /// Returning `None` if there are less than `n + 1` elements:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().nth(10), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        for x in self {
            if n == 0 { return Some(x) }
            n -= 1;
        }
        None
    }

    /// Creates an iterator starting at the same point, but stepping by
    /// the given amount at each iteration.
    ///
    /// Note 1: The first element of the iterator will always be returned,
    /// regardless of the step given.
    ///
    /// Note 2: The time at which ignored elements are pulled is not fixed.
    /// `StepBy` behaves like the sequence `next(), nth(step-1), nth(step-1), …`,
    /// but is also free to behave like the sequence
    /// `advance_n_and_return_first(step), advance_n_and_return_first(step), …`
    /// Which way is used may change for some iterators for performance reasons.
    /// The second way will advance the iterator earlier and may consume more items.
    ///
    /// `advance_n_and_return_first` is the equivalent of:
    /// ```
    /// fn advance_n_and_return_first<I>(iter: &mut I, total_step: usize) -> Option<I::Item>
    /// where
    ///     I: Iterator,
    /// {
    ///     let next = iter.next();
    ///     if total_step > 1 {
    ///         iter.nth(total_step-2);
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
    /// Basic usage:
    ///
    /// ```
    /// let a = [0, 1, 2, 3, 4, 5];
    /// let mut iter = a.iter().step_by(2);
    ///
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "iterator_step_by", since = "1.28.0")]
    fn step_by(self, step: usize) -> StepBy<Self> where Self: Sized {
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
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a1 = [1, 2, 3];
    /// let a2 = [4, 5, 6];
    ///
    /// let mut iter = a1.iter().chain(a2.iter());
    ///
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), Some(&5));
    /// assert_eq!(iter.next(), Some(&6));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Since the argument to `chain()` uses [`IntoIterator`], we can pass
    /// anything that can be converted into an [`Iterator`], not just an
    /// [`Iterator`] itself. For example, slices (`&[T]`) implement
    /// [`IntoIterator`], and so can be passed to `chain()` directly:
    ///
    /// [`IntoIterator`]: trait.IntoIterator.html
    /// [`Iterator`]: trait.Iterator.html
    ///
    /// ```
    /// let s1 = &[1, 2, 3];
    /// let s2 = &[4, 5, 6];
    ///
    /// let mut iter = s1.iter().chain(s2);
    ///
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), Some(&5));
    /// assert_eq!(iter.next(), Some(&6));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chain<U>(self, other: U) -> Chain<Self, U::IntoIter> where
        Self: Sized, U: IntoIterator<Item=Self::Item>,
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
    /// will return [`None`]. If the first iterator returns [`None`], `zip` will
    /// short-circuit and `next` will not be called on the second iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a1 = [1, 2, 3];
    /// let a2 = [4, 5, 6];
    ///
    /// let mut iter = a1.iter().zip(a2.iter());
    ///
    /// assert_eq!(iter.next(), Some((&1, &4)));
    /// assert_eq!(iter.next(), Some((&2, &5)));
    /// assert_eq!(iter.next(), Some((&3, &6)));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Since the argument to `zip()` uses [`IntoIterator`], we can pass
    /// anything that can be converted into an [`Iterator`], not just an
    /// [`Iterator`] itself. For example, slices (`&[T]`) implement
    /// [`IntoIterator`], and so can be passed to `zip()` directly:
    ///
    /// [`IntoIterator`]: trait.IntoIterator.html
    /// [`Iterator`]: trait.Iterator.html
    ///
    /// ```
    /// let s1 = &[1, 2, 3];
    /// let s2 = &[4, 5, 6];
    ///
    /// let mut iter = s1.iter().zip(s2);
    ///
    /// assert_eq!(iter.next(), Some((&1, &4)));
    /// assert_eq!(iter.next(), Some((&2, &5)));
    /// assert_eq!(iter.next(), Some((&3, &6)));
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
    /// [`enumerate`]: trait.Iterator.html#method.enumerate
    /// [`next`]: ../../std/iter/trait.Iterator.html#tymethod.next
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn zip<U>(self, other: U) -> Zip<Self, U::IntoIter> where
        Self: Sized, U: IntoIterator
    {
        Zip::new(self, other.into_iter())
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
    /// [`FnMut`]: ../../std/ops/trait.FnMut.html
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
    /// (0..5).map(|x| println!("{}", x));
    ///
    /// // it won't even execute, as it is lazy. Rust will warn you about this.
    ///
    /// // Instead, use for:
    /// for x in 0..5 {
    ///     println!("{}", x);
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn map<B, F>(self, f: F) -> Map<Self, F> where
        Self: Sized, F: FnMut(Self::Item) -> B,
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
    /// internal iteration on adaptors like `Chain`.
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
    /// let v: Vec<_> =  rx.iter().collect();
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
    ///       .for_each(|(i, x)| println!("{}:{}", i, x));
    /// ```
    #[inline]
    #[stable(feature = "iterator_for_each", since = "1.21.0")]
    fn for_each<F>(self, mut f: F) where
        Self: Sized, F: FnMut(Self::Item),
    {
        self.fold((), move |(), item| f(item));
    }

    /// Creates an iterator which uses a closure to determine if an element
    /// should be yielded.
    ///
    /// The closure must return `true` or `false`. `filter()` creates an
    /// iterator which calls this closure on each element. If the closure
    /// returns `true`, then the element is returned. If the closure returns
    /// `false`, it will try again, and call the closure on the next element,
    /// seeing if it passes the test.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [0i32, 1, 2];
    ///
    /// let mut iter = a.iter().filter(|x| x.is_positive());
    ///
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the closure passed to `filter()` takes a reference, and many
    /// iterators iterate over references, this leads to a possibly confusing
    /// situation, where the type of the closure is a double reference:
    ///
    /// ```
    /// let a = [0, 1, 2];
    ///
    /// let mut iter = a.iter().filter(|x| **x > 1); // need two *s!
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// It's common to instead use destructuring on the argument to strip away
    /// one:
    ///
    /// ```
    /// let a = [0, 1, 2];
    ///
    /// let mut iter = a.iter().filter(|&x| *x > 1); // both & and *
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// or both:
    ///
    /// ```
    /// let a = [0, 1, 2];
    ///
    /// let mut iter = a.iter().filter(|&&x| x > 1); // two &s
    ///
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// of these layers.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter<P>(self, predicate: P) -> Filter<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
    {
        Filter::new(self, predicate)
    }

    /// Creates an iterator that both filters and maps.
    ///
    /// The closure must return an [`Option<T>`]. `filter_map` creates an
    /// iterator which calls this closure on each element. If the closure
    /// returns [`Some(element)`][`Some`], then that element is returned. If the
    /// closure returns [`None`], it will try again, and call the closure on the
    /// next element, seeing if it will return [`Some`].
    ///
    /// Why `filter_map` and not just [`filter`] and [`map`]? The key is in this
    /// part:
    ///
    /// [`filter`]: #method.filter
    /// [`map`]: #method.map
    ///
    /// > If the closure returns [`Some(element)`][`Some`], then that element is returned.
    ///
    /// In other words, it removes the [`Option<T>`] layer automatically. If your
    /// mapping is already returning an [`Option<T>`] and you want to skip over
    /// [`None`]s, then `filter_map` is much, much nicer to use.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = ["1", "lol", "3", "NaN", "5"];
    ///
    /// let mut iter = a.iter().filter_map(|s| s.parse().ok());
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Here's the same example, but with [`filter`] and [`map`]:
    ///
    /// ```
    /// let a = ["1", "lol", "3", "NaN", "5"];
    /// let mut iter = a.iter().map(|s| s.parse()).filter(|s| s.is_ok()).map(|s| s.unwrap());
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// [`Option<T>`]: ../../std/option/enum.Option.html
    /// [`Some`]: ../../std/option/enum.Option.html#variant.Some
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F> where
        Self: Sized, F: FnMut(Self::Item) -> Option<B>,
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
    /// debug assertions are enabled, a panic is guaranteed.
    ///
    /// # Panics
    ///
    /// The returned iterator might panic if the to-be-returned index would
    /// overflow a [`usize`].
    ///
    /// [`usize::MAX`]: ../../std/usize/constant.MAX.html
    /// [`usize`]: ../../std/primitive.usize.html
    /// [`zip`]: #method.zip
    ///
    /// # Examples
    ///
    /// ```
    /// let a = ['a', 'b', 'c'];
    ///
    /// let mut iter = a.iter().enumerate();
    ///
    /// assert_eq!(iter.next(), Some((0, &'a')));
    /// assert_eq!(iter.next(), Some((1, &'b')));
    /// assert_eq!(iter.next(), Some((2, &'c')));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn enumerate(self) -> Enumerate<Self> where Self: Sized {
        Enumerate::new(self)
    }

    /// Creates an iterator which can use `peek` to look at the next element of
    /// the iterator without consuming it.
    ///
    /// Adds a [`peek`] method to an iterator. See its documentation for
    /// more information.
    ///
    /// Note that the underlying iterator is still advanced when [`peek`] is
    /// called for the first time: In order to retrieve the next element,
    /// [`next`] is called on the underlying iterator, hence any side effects (i.e.
    /// anything other than fetching the next value) of the [`next`] method
    /// will occur.
    ///
    /// [`peek`]: struct.Peekable.html#method.peek
    /// [`next`]: ../../std/iter/trait.Iterator.html#tymethod.next
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
    /// // we can peek() multiple times, the iterator won't advance
    /// assert_eq!(iter.peek(), Some(&&3));
    /// assert_eq!(iter.peek(), Some(&&3));
    ///
    /// assert_eq!(iter.next(), Some(&3));
    ///
    /// // after the iterator is finished, so is peek()
    /// assert_eq!(iter.peek(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn peekable(self) -> Peekable<Self> where Self: Sized {
        Peekable::new(self)
    }

    /// Creates an iterator that [`skip`]s elements based on a predicate.
    ///
    /// [`skip`]: #method.skip
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
    /// let mut iter = a.iter().skip_while(|x| x.is_negative());
    ///
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the closure passed to `skip_while()` takes a reference, and many
    /// iterators iterate over references, this leads to a possibly confusing
    /// situation, where the type of the closure is a double reference:
    ///
    /// ```
    /// let a = [-1, 0, 1];
    ///
    /// let mut iter = a.iter().skip_while(|x| **x < 0); // need two *s!
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
    /// let mut iter = a.iter().skip_while(|x| **x < 0);
    ///
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// // while this would have been false, since we already got a false,
    /// // skip_while() isn't used any more
    /// assert_eq!(iter.next(), Some(&-2));
    ///
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
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
    /// let mut iter = a.iter().take_while(|x| x.is_negative());
    ///
    /// assert_eq!(iter.next(), Some(&-1));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because the closure passed to `take_while()` takes a reference, and many
    /// iterators iterate over references, this leads to a possibly confusing
    /// situation, where the type of the closure is a double reference:
    ///
    /// ```
    /// let a = [-1, 0, 1];
    ///
    /// let mut iter = a.iter().take_while(|x| **x < 0); // need two *s!
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
    /// let mut iter = a.iter().take_while(|x| **x < 0);
    ///
    /// assert_eq!(iter.next(), Some(&-1));
    ///
    /// // We have more elements that are less than zero, but since we already
    /// // got a false, take_while() isn't used any more
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Because `take_while()` needs to look at the value in order to see if it
    /// should be included or not, consuming iterators will see that it is
    /// removed:
    ///
    /// ```
    /// let a = [1, 2, 3, 4];
    /// let mut iter = a.iter();
    ///
    /// let result: Vec<i32> = iter.by_ref()
    ///                            .take_while(|n| **n != 3)
    ///                            .cloned()
    ///                            .collect();
    ///
    /// assert_eq!(result, &[1, 2]);
    ///
    /// let result: Vec<i32> = iter.cloned().collect();
    ///
    /// assert_eq!(result, &[4]);
    /// ```
    ///
    /// The `3` is no longer there, because it was consumed in order to see if
    /// the iteration should stop, but wasn't placed back into the iterator.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
    {
        TakeWhile::new(self, predicate)
    }

    /// Creates an iterator that skips the first `n` elements.
    ///
    /// After they have been consumed, the rest of the elements are yielded.
    /// Rather than overriding this method directly, instead override the `nth` method.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter().skip(2);
    ///
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn skip(self, n: usize) -> Skip<Self> where Self: Sized {
        Skip::new(self, n)
    }

    /// Creates an iterator that yields its first `n` elements.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter().take(2);
    ///
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take(self, n: usize) -> Take<Self> where Self: Sized, {
        Take::new(self, n)
    }

    /// An iterator adaptor similar to [`fold`] that holds internal state and
    /// produces a new iterator.
    ///
    /// [`fold`]: #method.fold
    ///
    /// `scan()` takes two arguments: an initial value which seeds the internal
    /// state, and a closure with two arguments, the first being a mutable
    /// reference to the internal state and the second an iterator element.
    /// The closure can assign to the internal state to share state between
    /// iterations.
    ///
    /// On iteration, the closure will be applied to each element of the
    /// iterator and the return value from the closure, an [`Option`], is
    /// yielded by the iterator.
    ///
    /// [`Option`]: ../../std/option/enum.Option.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter().scan(1, |state, &x| {
    ///     // each iteration, we'll multiply the state by the element
    ///     *state = *state * x;
    ///
    ///     // then, we'll yield the negation of the state
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
        where Self: Sized, F: FnMut(&mut St, Self::Item) -> Option<B>,
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
    /// [`map`]: #method.map
    /// [`flatten`]: #method.flatten
    ///
    /// # Examples
    ///
    /// Basic usage:
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
        where Self: Sized, U: IntoIterator, F: FnMut(Self::Item) -> U,
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
    /// let flattened = data.into_iter().flatten().collect::<Vec<u8>>();
    /// assert_eq!(flattened, &[1, 2, 3, 4, 5, 6]);
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
    /// Flattening once only removes one level of nesting:
    ///
    /// ```
    /// let d3 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    ///
    /// let d2 = d3.iter().flatten().collect::<Vec<_>>();
    /// assert_eq!(d2, [&[1, 2], &[3, 4], &[5, 6], &[7, 8]]);
    ///
    /// let d1 = d3.iter().flatten().flatten().collect::<Vec<_>>();
    /// assert_eq!(d1, [&1, &2, &3, &4, &5, &6, &7, &8]);
    /// ```
    ///
    /// Here we see that `flatten()` does not perform a "deep" flatten.
    /// Instead, only one level of nesting is removed. That is, if you
    /// `flatten()` a three-dimensional array the result will be
    /// two-dimensional and not one-dimensional. To get a one-dimensional
    /// structure, you have to `flatten()` again.
    ///
    /// [`flat_map()`]: #method.flat_map
    #[inline]
    #[stable(feature = "iterator_flatten", since = "1.29.0")]
    fn flatten(self) -> Flatten<Self>
    where Self: Sized, Self::Item: IntoIterator {
        Flatten::new(self)
    }

    /// Creates an iterator which ends after the first [`None`].
    ///
    /// After an iterator returns [`None`], future calls may or may not yield
    /// [`Some(T)`] again. `fuse()` adapts an iterator, ensuring that after a
    /// [`None`] is given, it will always return [`None`] forever.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`Some(T)`]: ../../std/option/enum.Option.html#variant.Some
    ///
    /// # Examples
    ///
    /// Basic usage:
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
    ///         if val % 2 == 0 {
    ///             Some(val)
    ///         } else {
    ///             None
    ///         }
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
    fn fuse(self) -> Fuse<Self> where Self: Sized {
        Fuse::new(self)
    }

    /// Do something with each element of an iterator, passing the value on.
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
    /// println!("{}", sum);
    ///
    /// // let's add some inspect() calls to investigate what's happening
    /// let sum = a.iter()
    ///     .cloned()
    ///     .inspect(|x| println!("about to filter: {}", x))
    ///     .filter(|x| x % 2 == 0)
    ///     .inspect(|x| println!("made it through filter: {}", x))
    ///     .fold(0, |sum, i| sum + i);
    ///
    /// println!("{}", sum);
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
    ///             println!("Parsing error: {}", e);
    ///         }
    ///     })
    ///     .filter_map(Result::ok)
    ///     .sum();
    ///
    /// println!("Sum: {}", sum);
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
    fn inspect<F>(self, f: F) -> Inspect<Self, F> where
        Self: Sized, F: FnMut(&Self::Item),
    {
        Inspect::new(self, f)
    }

    /// Borrows an iterator, rather than consuming it.
    ///
    /// This is useful to allow applying iterator adaptors while still
    /// retaining ownership of the original iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let iter = a.iter();
    ///
    /// let sum: i32 = iter.take(5).fold(0, |acc, i| acc + i );
    ///
    /// assert_eq!(sum, 6);
    ///
    /// // if we try to use iter again, it won't work. The following line
    /// // gives "error: use of moved value: `iter`
    /// // assert_eq!(iter.next(), None);
    ///
    /// // let's try that again
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// // instead, we add in a .by_ref()
    /// let sum: i32 = iter.by_ref().take(2).fold(0, |acc, i| acc + i );
    ///
    /// assert_eq!(sum, 3);
    ///
    /// // now this is just fine:
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn by_ref(&mut self) -> &mut Self where Self: Sized { self }

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
    /// One of the keys to `collect()`'s power is that many things you might
    /// not think of as 'collections' actually are. For example, a [`String`]
    /// is a collection of [`char`]s. And a collection of
    /// [`Result<T, E>`][`Result`] can be thought of as single
    /// [`Result`]`<Collection<T>, E>`. See the examples below for more.
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
    ///                          .map(|&x| x * 2)
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
    /// let doubled: VecDeque<i32> = a.iter().map(|&x| x * 2).collect();
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
    /// let hello: String = chars.iter()
    ///     .map(|&x| x as u8)
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
    /// let result: Result<Vec<_>, &str> = results.iter().cloned().collect();
    ///
    /// // gives us the first error
    /// assert_eq!(Err("nope"), result);
    ///
    /// let results = [Ok(1), Ok(3)];
    ///
    /// let result: Result<Vec<_>, &str> = results.iter().cloned().collect();
    ///
    /// // gives us the list of answers
    /// assert_eq!(Ok(vec![1, 3]), result);
    /// ```
    ///
    /// [`iter`]: ../../std/iter/trait.Iterator.html#tymethod.next
    /// [`String`]: ../../std/string/struct.String.html
    /// [`char`]: ../../std/primitive.char.html
    /// [`Result`]: ../../std/result/enum.Result.html
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use = "if you really need to exhaust the iterator, consider `.for_each(drop)` instead"]
    fn collect<B: FromIterator<Self::Item>>(self) -> B where Self: Sized {
        FromIterator::from_iter(self)
    }

    /// Consumes an iterator, creating two collections from it.
    ///
    /// The predicate passed to `partition()` can return `true`, or `false`.
    /// `partition()` returns a pair, all of the elements for which it returned
    /// `true`, and all of the elements for which it returned `false`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let (even, odd): (Vec<i32>, Vec<i32>) = a
    ///     .iter()
    ///     .partition(|&n| n % 2 == 0);
    ///
    /// assert_eq!(even, vec![2]);
    /// assert_eq!(odd, vec![1, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn partition<B, F>(self, mut f: F) -> (B, B) where
        Self: Sized,
        B: Default + Extend<Self::Item>,
        F: FnMut(&Self::Item) -> bool
    {
        let mut left: B = Default::default();
        let mut right: B = Default::default();

        self.for_each(|x| {
            if f(&x) {
                left.extend(Some(x))
            } else {
                right.extend(Some(x))
            }
        });

        (left, right)
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
    /// Most of the other (forward) methods have default implementations in
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
    /// let sum = a.iter().try_fold(0i8, |acc, &x| acc.checked_add(x));
    ///
    /// assert_eq!(sum, Some(6));
    /// ```
    ///
    /// Short-circuiting:
    ///
    /// ```
    /// let a = [10, 20, 30, 100, 40, 50];
    /// let mut it = a.iter();
    ///
    /// // This sum overflows when adding the 100 element
    /// let sum = it.try_fold(0i8, |acc, &x| acc.checked_add(x));
    /// assert_eq!(sum, None);
    ///
    /// // Because it short-circuited, the remaining elements are still
    /// // available through the iterator.
    /// assert_eq!(it.len(), 2);
    /// assert_eq!(it.next(), Some(&40));
    /// ```
    #[inline]
    #[stable(feature = "iterator_try_fold", since = "1.27.0")]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R where
        Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x)?;
        }
        Try::from_ok(accum)
    }

    /// An iterator method that applies a fallible function to each item in the
    /// iterator, stopping at the first error and returning that error.
    ///
    /// This can also be thought of as the fallible form of [`for_each()`]
    /// or as the stateless version of [`try_fold()`].
    ///
    /// [`for_each()`]: #method.for_each
    /// [`try_fold()`]: #method.try_fold
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
    /// let res = data.iter().try_for_each(|x| writeln!(stdout(), "{}", x));
    /// assert!(res.is_ok());
    ///
    /// let mut it = data.iter().cloned();
    /// let res = it.try_for_each(|x| rename(x, Path::new(x).with_extension("old")));
    /// assert!(res.is_err());
    /// // It short-circuited, so the remaining items are still in the iterator:
    /// assert_eq!(it.next(), Some("stale_bread.json"));
    /// ```
    #[inline]
    #[stable(feature = "iterator_try_fold", since = "1.27.0")]
    fn try_for_each<F, R>(&mut self, mut f: F) -> R where
        Self: Sized, F: FnMut(Self::Item) -> R, R: Try<Ok=()>
    {
        self.try_fold((), move |(), x| f(x))
    }

    /// An iterator method that applies a function, producing a single, final value.
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
    /// may not terminate for infinite iterators, even on traits for which a
    /// result is determinable in finite time.
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
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B where
        Self: Sized, F: FnMut(B, Self::Item) -> B,
    {
        self.try_fold(init, move |acc, x| Ok::<B, !>(f(acc, x))).unwrap()
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
    /// assert!(a.iter().all(|&x| x > 0));
    ///
    /// assert!(!a.iter().all(|&x| x > 2));
    /// ```
    ///
    /// Stopping at the first `false`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert!(!iter.all(|&x| x != 2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(&3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn all<F>(&mut self, mut f: F) -> bool where
        Self: Sized, F: FnMut(Self::Item) -> bool
    {
        self.try_for_each(move |x| {
            if f(x) { LoopState::Continue(()) }
            else { LoopState::Break(()) }
        }) == LoopState::Continue(())
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
    /// assert!(a.iter().any(|&x| x > 0));
    ///
    /// assert!(!a.iter().any(|&x| x > 5));
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert!(iter.any(|&x| x != 2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(&2));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn any<F>(&mut self, mut f: F) -> bool where
        Self: Sized,
        F: FnMut(Self::Item) -> bool
    {
        self.try_for_each(move |x| {
            if f(x) { LoopState::Break(()) }
            else { LoopState::Continue(()) }
        }) == LoopState::Break(())
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
    /// assert_eq!(a.iter().find(|&&x| x == 2), Some(&2));
    ///
    /// assert_eq!(a.iter().find(|&&x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.find(|&&x| x == 2), Some(&2));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(&3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item> where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        self.try_for_each(move |x| {
            if predicate(&x) { LoopState::Break(x) }
            else { LoopState::Continue(()) }
        }).break_value()
    }

    /// Applies function to the elements of iterator and returns
    /// the first non-none result.
    ///
    /// `iter.find_map(f)` is equivalent to `iter.filter_map(f).next()`.
    ///
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
    fn find_map<B, F>(&mut self, mut f: F) -> Option<B> where
        Self: Sized,
        F: FnMut(Self::Item) -> Option<B>,
    {
        self.try_for_each(move |x| {
            match f(x) {
                Some(x) => LoopState::Break(x),
                None => LoopState::Continue(()),
            }
        }).break_value()
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
    /// result or panics. If debug assertions are enabled, a panic is
    /// guaranteed.
    ///
    /// # Panics
    ///
    /// This function might panic if the iterator has more than `usize::MAX`
    /// non-matching elements.
    ///
    /// [`Some(index)`]: ../../std/option/enum.Option.html#variant.Some
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    /// [`usize::MAX`]: ../../std/usize/constant.MAX.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.iter().position(|&x| x == 2), Some(1));
    ///
    /// assert_eq!(a.iter().position(|&x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3, 4];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.position(|&x| x >= 2), Some(1));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(&3));
    ///
    /// // The returned index depends on iterator state
    /// assert_eq!(iter.position(|&x| x == 4), Some(0));
    ///
    /// ```
    #[inline]
    #[rustc_inherit_overflow_checks]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        // The addition might panic on overflow
        self.try_fold(0, move |i, x| {
            if predicate(x) { LoopState::Break(i) }
            else { LoopState::Continue(i + 1) }
        }).break_value()
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
    /// [`Some(index)`]: ../../std/option/enum.Option.html#variant.Some
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.iter().rposition(|&x| x == 3), Some(2));
    ///
    /// assert_eq!(a.iter().rposition(|&x| x == 5), None);
    /// ```
    ///
    /// Stopping at the first `true`:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.rposition(|&x| x == 2), Some(1));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(&1));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
        P: FnMut(Self::Item) -> bool,
        Self: Sized + ExactSizeIterator + DoubleEndedIterator
    {
        // No need for an overflow check here, because `ExactSizeIterator`
        // implies that the number of elements fits into a `usize`.
        let n = self.len();
        self.try_rfold(n, move |i, x| {
            let i = i - 1;
            if predicate(x) { LoopState::Break(i) }
            else { LoopState::Continue(i) }
        }).break_value()
    }

    /// Returns the maximum element of an iterator.
    ///
    /// If several elements are equally maximum, the last element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let b: Vec<u32> = Vec::new();
    ///
    /// assert_eq!(a.iter().max(), Some(&3));
    /// assert_eq!(b.iter().max(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn max(self) -> Option<Self::Item> where Self: Sized, Self::Item: Ord
    {
        self.max_by(Ord::cmp)
    }

    /// Returns the minimum element of an iterator.
    ///
    /// If several elements are equally minimum, the first element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let b: Vec<u32> = Vec::new();
    ///
    /// assert_eq!(a.iter().min(), Some(&1));
    /// assert_eq!(b.iter().min(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn min(self) -> Option<Self::Item> where Self: Sized, Self::Item: Ord
    {
        self.min_by(Ord::cmp)
    }

    /// Returns the element that gives the maximum value from the
    /// specified function.
    ///
    /// If several elements are equally maximum, the last element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().max_by_key(|x| x.abs()).unwrap(), -10);
    /// ```
    #[inline]
    #[stable(feature = "iter_cmp_by_key", since = "1.6.0")]
    fn max_by_key<B: Ord, F>(self, mut f: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item) -> B,
    {
        // switch to y even if it is only equal, to preserve stability.
        select_fold1(self.map(|x| (f(&x), x)), |(x_p, _), (y_p, _)| x_p <= y_p).map(|(_, x)| x)
    }

    /// Returns the element that gives the maximum value with respect to the
    /// specified comparison function.
    ///
    /// If several elements are equally maximum, the last element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().max_by(|x, y| x.cmp(y)).unwrap(), 5);
    /// ```
    #[inline]
    #[stable(feature = "iter_max_by", since = "1.15.0")]
    fn max_by<F>(self, mut compare: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        // switch to y even if it is only equal, to preserve stability.
        select_fold1(self, |x, y| compare(x, y) != Ordering::Greater)
    }

    /// Returns the element that gives the minimum value from the
    /// specified function.
    ///
    /// If several elements are equally minimum, the first element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().min_by_key(|x| x.abs()).unwrap(), 0);
    /// ```
    #[stable(feature = "iter_cmp_by_key", since = "1.6.0")]
    fn min_by_key<B: Ord, F>(self, mut f: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item) -> B,
    {
        // only switch to y if it is strictly smaller, to preserve stability.
        select_fold1(self.map(|x| (f(&x), x)), |(x_p, _), (y_p, _)| x_p > y_p).map(|(_, x)| x)
    }

    /// Returns the element that gives the minimum value with respect to the
    /// specified comparison function.
    ///
    /// If several elements are equally minimum, the first element is
    /// returned. If the iterator is empty, [`None`] is returned.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().min_by(|x, y| x.cmp(y)).unwrap(), -10);
    /// ```
    #[inline]
    #[stable(feature = "iter_min_by", since = "1.15.0")]
    fn min_by<F>(self, mut compare: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        // only switch to y if it is strictly smaller, to preserve stability.
        select_fold1(self, |x, y| compare(x, y) == Ordering::Greater)
    }


    /// Reverses an iterator's direction.
    ///
    /// Usually, iterators iterate from left to right. After using `rev()`,
    /// an iterator will instead iterate from right to left.
    ///
    /// This is only possible if the iterator has an end, so `rev()` only
    /// works on [`DoubleEndedIterator`]s.
    ///
    /// [`DoubleEndedIterator`]: trait.DoubleEndedIterator.html
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter().rev();
    ///
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rev(self) -> Rev<Self> where Self: Sized + DoubleEndedIterator {
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
    /// [`zip`]: #method.zip
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [(1, 2), (3, 4)];
    ///
    /// let (left, right): (Vec<_>, Vec<_>) = a.iter().cloned().unzip();
    ///
    /// assert_eq!(left, [1, 3]);
    /// assert_eq!(right, [2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB) where
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
        Self: Sized + Iterator<Item=(A, B)>,
    {
        let mut ts: FromA = Default::default();
        let mut us: FromB = Default::default();

        self.for_each(|(t, u)| {
            ts.extend(Some(t));
            us.extend(Some(u));
        });

        (ts, us)
    }

    /// Creates an iterator which copies all of its elements.
    ///
    /// This is useful when you have an iterator over `&T`, but you need an
    /// iterator over `T`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let v_cloned: Vec<_> = a.iter().copied().collect();
    ///
    /// // copied is the same as .map(|&x| x)
    /// let v_map: Vec<_> = a.iter().map(|&x| x).collect();
    ///
    /// assert_eq!(v_cloned, vec![1, 2, 3]);
    /// assert_eq!(v_map, vec![1, 2, 3]);
    /// ```
    #[stable(feature = "iter_copied", since = "1.36.0")]
    fn copied<'a, T: 'a>(self) -> Copied<Self>
        where Self: Sized + Iterator<Item=&'a T>, T: Copy
    {
        Copied::new(self)
    }

    /// Creates an iterator which [`clone`]s all of its elements.
    ///
    /// This is useful when you have an iterator over `&T`, but you need an
    /// iterator over `T`.
    ///
    /// [`clone`]: ../../std/clone/trait.Clone.html#tymethod.clone
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
    /// assert_eq!(v_cloned, vec![1, 2, 3]);
    /// assert_eq!(v_map, vec![1, 2, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn cloned<'a, T: 'a>(self) -> Cloned<Self>
        where Self: Sized + Iterator<Item=&'a T>, T: Clone
    {
        Cloned::new(self)
    }

    /// Repeats an iterator endlessly.
    ///
    /// Instead of stopping at [`None`], the iterator will instead start again,
    /// from the beginning. After iterating again, it will start at the
    /// beginning again. And again. And again. Forever.
    ///
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut it = a.iter().cycle();
    ///
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert_eq!(it.next(), Some(&3));
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert_eq!(it.next(), Some(&3));
    /// assert_eq!(it.next(), Some(&1));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn cycle(self) -> Cycle<Self> where Self: Sized + Clone {
        Cycle::new(self)
    }

    /// Sums the elements of an iterator.
    ///
    /// Takes each element, adds them together, and returns the result.
    ///
    /// An empty iterator returns the zero value of the type.
    ///
    /// # Panics
    ///
    /// When calling `sum()` and a primitive integer type is being returned, this
    /// method will panic if the computation overflows and debug assertions are
    /// enabled.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// let sum: i32 = a.iter().sum();
    ///
    /// assert_eq!(sum, 6);
    /// ```
    #[stable(feature = "iter_arith", since = "1.11.0")]
    fn sum<S>(self) -> S
        where Self: Sized,
              S: Sum<Self::Item>,
    {
        Sum::sum(self)
    }

    /// Iterates over the entire iterator, multiplying all the elements
    ///
    /// An empty iterator returns the one value of the type.
    ///
    /// # Panics
    ///
    /// When calling `product()` and a primitive integer type is being returned,
    /// method will panic if the computation overflows and debug assertions are
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
        where Self: Sized,
              P: Product<Self::Item>,
    {
        Product::product(self)
    }

    /// Lexicographically compares the elements of this `Iterator` with those
    /// of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn cmp<I>(mut self, other: I) -> Ordering where
        I: IntoIterator<Item = Self::Item>,
        Self::Item: Ord,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            let x = match self.next() {
                None => if other.next().is_none() {
                    return Ordering::Equal
                } else {
                    return Ordering::Less
                },
                Some(val) => val,
            };

            let y = match other.next() {
                None => return Ordering::Greater,
                Some(val) => val,
            };

            match x.cmp(&y) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            }
        }
    }

    /// Lexicographically compares the elements of this `Iterator` with those
    /// of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn partial_cmp<I>(mut self, other: I) -> Option<Ordering> where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            let x = match self.next() {
                None => if other.next().is_none() {
                    return Some(Ordering::Equal)
                } else {
                    return Some(Ordering::Less)
                },
                Some(val) => val,
            };

            let y = match other.next() {
                None => return Some(Ordering::Greater),
                Some(val) => val,
            };

            match x.partial_cmp(&y) {
                Some(Ordering::Equal) => (),
                non_eq => return non_eq,
            }
        }
    }

    /// Determines if the elements of this `Iterator` are equal to those of
    /// another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn eq<I>(mut self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            let x = match self.next() {
                None => return other.next().is_none(),
                Some(val) => val,
            };

            let y = match other.next() {
                None => return false,
                Some(val) => val,
            };

            if x != y { return false }
        }
    }

    /// Determines if the elements of this `Iterator` are unequal to those of
    /// another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn ne<I>(self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
        Self: Sized,
    {
        !self.eq(other)
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// less than those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn lt<I>(self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        self.partial_cmp(other) == Some(Ordering::Less)
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// less or equal to those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn le<I>(self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        match self.partial_cmp(other) {
            Some(Ordering::Less) | Some(Ordering::Equal) => true,
            _ => false,
        }
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// greater than those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn gt<I>(self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        self.partial_cmp(other) == Some(Ordering::Greater)
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// greater than or equal to those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn ge<I>(self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        match self.partial_cmp(other) {
            Some(Ordering::Greater) | Some(Ordering::Equal) => true,
            _ => false,
        }
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
    /// #![feature(is_sorted)]
    ///
    /// assert!([1, 2, 2, 9].iter().is_sorted());
    /// assert!(![1, 3, 2, 4].iter().is_sorted());
    /// assert!([0].iter().is_sorted());
    /// assert!(std::iter::empty::<i32>().is_sorted());
    /// assert!(![0.0, 1.0, std::f32::NAN].iter().is_sorted());
    /// ```
    #[inline]
    #[unstable(feature = "is_sorted", reason = "new API", issue = "53485")]
    fn is_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        self.is_sorted_by(|a, b| a.partial_cmp(b))
    }

    /// Checks if the elements of this iterator are sorted using the given comparator function.
    ///
    /// Instead of using `PartialOrd::partial_cmp`, this function uses the given `compare`
    /// function to determine the ordering of two elements. Apart from that, it's equivalent to
    /// [`is_sorted`]; see its documentation for more information.
    ///
    /// [`is_sorted`]: trait.Iterator.html#method.is_sorted
    #[unstable(feature = "is_sorted", reason = "new API", issue = "53485")]
    fn is_sorted_by<F>(mut self, mut compare: F) -> bool
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>
    {
        let mut last = match self.next() {
            Some(e) => e,
            None => return true,
        };

        while let Some(curr) = self.next() {
            if compare(&last, &curr)
                .map(|o| o == Ordering::Greater)
                .unwrap_or(true)
            {
                return false;
            }
            last = curr;
        }

        true
    }

    /// Checks if the elements of this iterator are sorted using the given key extraction
    /// function.
    ///
    /// Instead of comparing the iterator's elements directly, this function compares the keys of
    /// the elements, as determined by `f`. Apart from that, it's equivalent to [`is_sorted`]; see
    /// its documentation for more information.
    ///
    /// [`is_sorted`]: trait.Iterator.html#method.is_sorted
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(is_sorted)]
    ///
    /// assert!(["c", "bb", "aaa"].iter().is_sorted_by_key(|s| s.len()));
    /// assert!(![-2i32, -1, 0, 3].iter().is_sorted_by_key(|n| n.abs()));
    /// ```
    #[inline]
    #[unstable(feature = "is_sorted", reason = "new API", issue = "53485")]
    fn is_sorted_by_key<F, K>(self, mut f: F) -> bool
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> K,
        K: PartialOrd
    {
        self.is_sorted_by(|a, b| f(a).partial_cmp(&f(b)))
    }
}

/// Select an element from an iterator based on the given "comparison"
/// function.
///
/// This is an idiosyncratic helper to try to factor out the
/// commonalities of {max,min}{,_by}. In particular, this avoids
/// having to implement optimizations several times.
#[inline]
fn select_fold1<I, F>(mut it: I, mut f: F) -> Option<I::Item>
    where
        I: Iterator,
        F: FnMut(&I::Item, &I::Item) -> bool,
{
    // start with the first element as our selection. This avoids
    // having to use `Option`s inside the loop, translating to a
    // sizeable performance gain (6x in one case).
    it.next().map(|first| {
        it.fold(first, |sel, x| if f(&sel, &x) { x } else { sel })
    })
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized> Iterator for &mut I {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> { (**self).next() }
    fn size_hint(&self) -> (usize, Option<usize>) { (**self).size_hint() }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth(n)
    }
}
