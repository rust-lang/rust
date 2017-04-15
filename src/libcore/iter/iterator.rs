// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp::Ordering;

use super::{Chain, Cycle, Cloned, Enumerate, Filter, FilterMap, FlatMap, Fuse};
use super::{Inspect, Map, Peekable, Scan, Skip, SkipWhile, Take, TakeWhile, Rev};
use super::{Zip, Sum, Product};
use super::{ChainState, FromIterator, ZipImpl};

fn _assert_is_object_safe(_: &Iterator<Item=()>) {}

/// An interface for dealing with iterators.
///
/// This is the main iterator trait. For more about the concept of iterators
/// generally, please see the [module-level documentation]. In particular, you
/// may want to know how to [implement `Iterator`][impl].
///
/// [module-level documentation]: index.html
/// [impl]: index.html#implementing-iterator
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "`{Self}` is not an iterator; maybe try calling \
                            `.iter()` or a similar method"]
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
    /// // More calls may or may not return None. Here, they always will.
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
    /// trusted to e.g. omit bounds checks in unsafe code. An incorrect
    /// implementation of `size_hint()` should not lead to memory safety
    /// violations.
    ///
    /// That said, the implementation should provide a correct estimation,
    /// because otherwise it would be a violation of the trait's protocol.
    ///
    /// The default implementation returns `(0, None)` which is correct for any
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
    /// // Let's add one five more numbers with chain()
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
    /// let iter = 0..;
    ///
    /// assert_eq!((0, None), iter.size_hint());
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
    /// [`usize::MAX`]: ../../std/isize/constant.MAX.html
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
        Chain{a: self, b: other.into_iter(), state: ChainState::Both}
    }

    /// 'Zips up' two iterators into a single iterator of pairs.
    ///
    /// `zip()` returns a new iterator that will iterate over two other
    /// iterators, returning a tuple where the first element comes from the
    /// first iterator, and the second element comes from the second iterator.
    ///
    /// In other words, it zips two iterators together, into a single one.
    ///
    /// When either iterator returns [`None`], all further calls to [`next`]
    /// will return [`None`].
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
    /// something that implements `FnMut`. It produces a new iterator which
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
    /// [`for`]: ../../book/first-edition/loops.html#for
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.into_iter().map(|x| 2 * x);
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
        Map{iter: self, f: f}
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
    /// let mut iter = a.into_iter().filter(|x| x.is_positive());
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
    /// let mut iter = a.into_iter().filter(|x| **x > 1); // need two *s!
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
    /// let mut iter = a.into_iter().filter(|&x| *x > 1); // both & and *
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
    /// let mut iter = a.into_iter().filter(|&&x| x > 1); // two &s
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
        Filter{iter: self, predicate: predicate}
    }

    /// Creates an iterator that both filters and maps.
    ///
    /// The closure must return an [`Option<T>`]. `filter_map` creates an
    /// iterator which calls this closure on each element. If the closure
    /// returns [`Some(element)`][`Some`], then that element is returned. If the
    /// closure returns [`None`], it will try again, and call the closure on the
    /// next element, seeing if it will return [`Some`].
    ///
    /// Why `filter_map` and not just [`filter`].[`map`]? The key is in this
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
    /// let a = ["1", "2", "lol"];
    ///
    /// let mut iter = a.iter().filter_map(|s| s.parse().ok());
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// Here's the same example, but with [`filter`] and [`map`]:
    ///
    /// ```
    /// let a = ["1", "2", "lol"];
    ///
    /// let mut iter = a.iter()
    ///                 .map(|s| s.parse().ok())
    ///                 .filter(|s| s.is_some());
    ///
    /// assert_eq!(iter.next(), Some(Some(1)));
    /// assert_eq!(iter.next(), Some(Some(2)));
    /// assert_eq!(iter.next(), None);
    /// ```
    ///
    /// There's an extra layer of [`Some`] in there.
    ///
    /// [`Option<T>`]: ../../std/option/enum.Option.html
    /// [`Some`]: ../../std/option/enum.Option.html#variant.Some
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F> where
        Self: Sized, F: FnMut(Self::Item) -> Option<B>,
    {
        FilterMap { iter: self, f: f }
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
        Enumerate { iter: self, count: 0 }
    }

    /// Creates an iterator which can use `peek` to look at the next element of
    /// the iterator without consuming it.
    ///
    /// Adds a [`peek`] method to an iterator. See its documentation for
    /// more information.
    ///
    /// Note that the underlying iterator is still advanced when [`peek`] is
    /// called for the first time: In order to retrieve the next element,
    /// [`next`] is called on the underlying iterator, hence any side effects of
    /// the [`next`] method will occur.
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
        Peekable{iter: self, peeked: None}
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
    /// let mut iter = a.into_iter().skip_while(|x| x.is_negative());
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
    /// let mut iter = a.into_iter().skip_while(|x| **x < 0); // need two *s!
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
    /// let mut iter = a.into_iter().skip_while(|x| **x < 0);
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
        SkipWhile{iter: self, flag: false, predicate: predicate}
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
    /// let mut iter = a.into_iter().take_while(|x| **x < 0); // need two *s!
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
    /// let mut iter = a.into_iter().take_while(|x| **x < 0);
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
    /// let mut iter = a.into_iter();
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
    /// the iteration should stop, but wasn't placed back into the iterator or
    /// some similar thing.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
    {
        TakeWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator that skips the first `n` elements.
    ///
    /// After they have been consumed, the rest of the elements are yielded.
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
        Skip{iter: self, n: n}
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
        Take{iter: self, n: n}
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
    ///     // the value passed on to the next iteration
    ///     Some(*state)
    /// });
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
        where Self: Sized, F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        Scan{iter: self, f: f, state: initial_state}
    }

    /// Creates an iterator that works like map, but flattens nested structure.
    ///
    /// The [`map`] adapter is very useful, but only when the closure
    /// argument produces values. If it produces an iterator instead, there's
    /// an extra layer of indirection. `flat_map()` will remove this extra layer
    /// on its own.
    ///
    /// Another way of thinking about `flat_map()`: [`map`]'s closure returns
    /// one item for each element, and `flat_map()`'s closure returns an
    /// iterator for each element.
    ///
    /// [`map`]: #method.map
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
        FlatMap{iter: self, f: f, frontiter: None, backiter: None }
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
    /// // it will always return None after the first time.
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fuse(self) -> Fuse<Self> where Self: Sized {
        Fuse{iter: self, done: false}
    }

    /// Do something with each element of an iterator, passing the value on.
    ///
    /// When using iterators, you'll often chain several of them together.
    /// While working on such code, you might want to check out what's
    /// happening at various parts in the pipeline. To do that, insert
    /// a call to `inspect()`.
    ///
    /// It's much more common for `inspect()` to be used as a debugging tool
    /// than to exist in your final code, but never say never.
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
    ///             .cloned()
    ///             .filter(|&x| x % 2 == 0)
    ///             .fold(0, |sum, i| sum + i);
    ///
    /// println!("{}", sum);
    ///
    /// // let's add some inspect() calls to investigate what's happening
    /// let sum = a.iter()
    ///             .cloned()
    ///             .inspect(|x| println!("about to filter: {}", x))
    ///             .filter(|&x| x % 2 == 0)
    ///             .inspect(|x| println!("made it through filter: {}", x))
    ///             .fold(0, |sum, i| sum + i);
    ///
    /// println!("{}", sum);
    /// ```
    ///
    /// This will print:
    ///
    /// ```text
    /// about to filter: 1
    /// about to filter: 4
    /// made it through filter: 4
    /// about to filter: 2
    /// made it through filter: 2
    /// about to filter: 3
    /// 6
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn inspect<F>(self, f: F) -> Inspect<Self, F> where
        Self: Sized, F: FnMut(&Self::Item),
    {
        Inspect{iter: self, f: f}
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
    /// let iter = a.into_iter();
    ///
    /// let sum: i32 = iter.take(5)
    ///                    .fold(0, |acc, &i| acc + i );
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
    /// let mut iter = a.into_iter();
    ///
    /// // instead, we add in a .by_ref()
    /// let sum: i32 = iter.by_ref()
    ///                    .take(2)
    ///                    .fold(0, |acc, &i| acc + i );
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
    /// let doubled: VecDeque<i32> = a.iter()
    ///                               .map(|&x| x * 2)
    ///                               .collect();
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
    /// let doubled = a.iter()
    ///                .map(|&x| x * 2)
    ///                .collect::<Vec<i32>>();
    ///
    /// assert_eq!(vec![2, 4, 6], doubled);
    /// ```
    ///
    /// Because `collect()` cares about what you're collecting into, you can
    /// still use a partial type hint, `_`, with the turbofish:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let doubled = a.iter()
    ///                .map(|&x| x * 2)
    ///                .collect::<Vec<_>>();
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
    ///                          .map(|&x| x as u8)
    ///                          .map(|x| (x + 1) as char)
    ///                          .collect();
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
    /// let (even, odd): (Vec<i32>, Vec<i32>) = a.into_iter()
    ///                                          .partition(|&n| n % 2 == 0);
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

        for x in self {
            if f(&x) {
                left.extend(Some(x))
            } else {
                right.extend(Some(x))
            }
        }

        (left, right)
    }

    /// An iterator adaptor that applies a function, producing a single, final value.
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
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// // the sum of all of the elements of a
    /// let sum = a.iter()
    ///            .fold(0, |acc, &x| acc + x);
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
    /// [`for`]: ../../book/first-edition/loops.html#for
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
    fn fold<B, F>(self, init: B, mut f: F) -> B where
        Self: Sized, F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        for x in self {
            accum = f(accum, x);
        }
        accum
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
        for x in self {
            if !f(x) {
                return false;
            }
        }
        true
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
        for x in self {
            if f(x) {
                return true;
            }
        }
        false
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
        for x in self {
            if predicate(&x) { return Some(x) }
        }
        None
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
    #[stable(feature = "rust1", since = "1.0.0")]
    fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        // `enumerate` might overflow.
        for (i, x) in self.enumerate() {
            if predicate(x) {
                return Some(i);
            }
        }
        None
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
        let mut i = self.len();

        while let Some(v) = self.next_back() {
            // No need for an overflow check here, because `ExactSizeIterator`
            // implies that the number of elements fits into a `usize`.
            i -= 1;
            if predicate(v) {
                return Some(i);
            }
        }
        None
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
        select_fold1(self,
                     |_| (),
                     // switch to y even if it is only equal, to preserve
                     // stability.
                     |_, x, _, y| *x <= *y)
            .map(|(_, x)| x)
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
        select_fold1(self,
                     |_| (),
                     // only switch to y if it is strictly smaller, to
                     // preserve stability.
                     |_, x, _, y| *x > *y)
            .map(|(_, x)| x)
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
    fn max_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item) -> B,
    {
        select_fold1(self,
                     f,
                     // switch to y even if it is only equal, to preserve
                     // stability.
                     |x_p, _, y_p, _| x_p <= y_p)
            .map(|(_, x)| x)
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
        select_fold1(self,
                     |_| (),
                     // switch to y even if it is only equal, to preserve
                     // stability.
                     |_, x, _, y| Ordering::Greater != compare(x, y))
            .map(|(_, x)| x)
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
    fn min_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item) -> B,
    {
        select_fold1(self,
                     f,
                     // only switch to y if it is strictly smaller, to
                     // preserve stability.
                     |x_p, _, y_p, _| x_p > y_p)
            .map(|(_, x)| x)
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
        select_fold1(self,
                     |_| (),
                     // switch to y even if it is strictly smaller, to
                     // preserve stability.
                     |_, x, _, y| Ordering::Greater == compare(x, y))
            .map(|(_, x)| x)
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
        Rev{iter: self}
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

        for (t, u) in self {
            ts.extend(Some(t));
            us.extend(Some(u));
        }

        (ts, us)
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
        Cloned { it: self }
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
        Cycle{orig: self.clone(), iter: self}
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
    ///     (1..).take_while(|&i| i <= n).product()
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
            match (self.next(), other.next()) {
                (None, None) => return Ordering::Equal,
                (None, _   ) => return Ordering::Less,
                (_   , None) => return Ordering::Greater,
                (Some(x), Some(y)) => match x.cmp(&y) {
                    Ordering::Equal => (),
                    non_eq => return non_eq,
                },
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
            match (self.next(), other.next()) {
                (None, None) => return Some(Ordering::Equal),
                (None, _   ) => return Some(Ordering::Less),
                (_   , None) => return Some(Ordering::Greater),
                (Some(x), Some(y)) => match x.partial_cmp(&y) {
                    Some(Ordering::Equal) => (),
                    non_eq => return non_eq,
                },
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
            match (self.next(), other.next()) {
                (None, None) => return true,
                (None, _) | (_, None) => return false,
                (Some(x), Some(y)) => if x != y { return false },
            }
        }
    }

    /// Determines if the elements of this `Iterator` are unequal to those of
    /// another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn ne<I>(mut self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            match (self.next(), other.next()) {
                (None, None) => return false,
                (None, _) | (_, None) => return true,
                (Some(x), Some(y)) => if x.ne(&y) { return true },
            }
        }
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// less than those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn lt<I>(mut self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            match (self.next(), other.next()) {
                (None, None) => return false,
                (None, _   ) => return true,
                (_   , None) => return false,
                (Some(x), Some(y)) => {
                    match x.partial_cmp(&y) {
                        Some(Ordering::Less) => return true,
                        Some(Ordering::Equal) => {}
                        Some(Ordering::Greater) => return false,
                        None => return false,
                    }
                },
            }
        }
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// less or equal to those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn le<I>(mut self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            match (self.next(), other.next()) {
                (None, None) => return true,
                (None, _   ) => return true,
                (_   , None) => return false,
                (Some(x), Some(y)) => {
                    match x.partial_cmp(&y) {
                        Some(Ordering::Less) => return true,
                        Some(Ordering::Equal) => {}
                        Some(Ordering::Greater) => return false,
                        None => return false,
                    }
                },
            }
        }
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// greater than those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn gt<I>(mut self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            match (self.next(), other.next()) {
                (None, None) => return false,
                (None, _   ) => return false,
                (_   , None) => return true,
                (Some(x), Some(y)) => {
                    match x.partial_cmp(&y) {
                        Some(Ordering::Less) => return false,
                        Some(Ordering::Equal) => {}
                        Some(Ordering::Greater) => return true,
                        None => return false,
                    }
                }
            }
        }
    }

    /// Determines if the elements of this `Iterator` are lexicographically
    /// greater than or equal to those of another.
    #[stable(feature = "iter_order", since = "1.5.0")]
    fn ge<I>(mut self, other: I) -> bool where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        let mut other = other.into_iter();

        loop {
            match (self.next(), other.next()) {
                (None, None) => return true,
                (None, _   ) => return false,
                (_   , None) => return true,
                (Some(x), Some(y)) => {
                    match x.partial_cmp(&y) {
                        Some(Ordering::Less) => return false,
                        Some(Ordering::Equal) => {}
                        Some(Ordering::Greater) => return true,
                        None => return false,
                    }
                },
            }
        }
    }
}

/// Select an element from an iterator based on the given "projection"
/// and "comparison" function.
///
/// This is an idiosyncratic helper to try to factor out the
/// commonalities of {max,min}{,_by}. In particular, this avoids
/// having to implement optimizations several times.
#[inline]
fn select_fold1<I, B, FProj, FCmp>(mut it: I,
                                   mut f_proj: FProj,
                                   mut f_cmp: FCmp) -> Option<(B, I::Item)>
    where I: Iterator,
          FProj: FnMut(&I::Item) -> B,
          FCmp: FnMut(&B, &I::Item, &B, &I::Item) -> bool
{
    // start with the first element as our selection. This avoids
    // having to use `Option`s inside the loop, translating to a
    // sizeable performance gain (6x in one case).
    it.next().map(|mut sel| {
        let mut sel_p = f_proj(&sel);

        for x in it {
            let x_p = f_proj(&x);
            if f_cmp(&sel_p, &sel, &x_p, &x) {
                sel = x;
                sel_p = x_p;
            }
        }
        (sel_p, sel)
    })
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: Iterator + ?Sized> Iterator for &'a mut I {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> { (**self).next() }
    fn size_hint(&self) -> (usize, Option<usize>) { (**self).size_hint() }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        (**self).nth(n)
    }
}
