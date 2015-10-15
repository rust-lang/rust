// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Composable external iteration
//!
//! If you've found yourself with a collection of some kind, and needed to
//! perform an operation on the elements of said collection, you'll quickly run
//! into 'iterators'. Iterators are heavily used in idiomatic Rust code, so
//! it's worth becoming familiar with them.
//!
//! Before explaining more, let's talk about how this module is structured:
//!
//! # Organization
//!
//! This module is largely organized by type:
//!
//! * [Traits] are the core portion: these traits define what kind of iterators
//!   exist and what you can do with them. The methods of these traits are worth
//!   putting some extra study time into.
//! * [Functions] provide some helpful ways to create some basic iterators.
//! * [Structs] are often the return types of the various methods on this
//!   module's traits. You'll usually want to look at the method that creates
//!   the `struct`, rather than the `struct` itself. For more detail about why,
//!   see '[Implementing Iterator](#implementing-iterator)'.
//!
//! [Traits]: #traits
//! [Functions]: #functions
//! [Structs]: #structs
//!
//! That's it! Let's dig into iterators.
//!
//! # Iterator
//!
//! The heart and soul of this module is the [`Iterator`] trait. The core of
//! [`Iterator`] looks like this:
//!
//! ```
//! trait Iterator {
//!     type Item;
//!     fn next(&mut self) -> Option<Self::Item>;
//! }
//! ```
//!
//! An iterator has a method, [`next()`], which when called, returns an
//! [`Option`]`<Item>`. [`next()`] will return `Some(Item)` as long as there
//! are elements, and once they've all been exhausted, will return `None` to
//! indicate that iteration is finished. Individual iterators may choose to
//! resume iteration, and so calling [`next()`] again may or may not eventually
//! start returning `Some(Item)` again at some point.
//!
//! [`Iterator`]'s full definition includes a number of other methods as well,
//! but they are default methods, built on top of [`next()`], and so you get
//! them for free.
//!
//! Iterators are also composable, and it's common to chain them together to do
//! more complex forms of processing. See the [Adapters](#adapters) section
//! below for more details.
//!
//! [`Iterator`]: trait.Iterator.html
//! [`next()`]: trait.Iterator.html#tymethod.next
//! [`Option`]: ../option/enum.Option.html
//!
//! # The three forms of iteration
//!
//! There are three common methods which can create iterators from a collection:
//!
//! * `iter()`, which iterates over `&T`.
//! * `iter_mut()`, which iterates over `&mut T`.
//! * `into_iter()`, which iterates over `T`.
//!
//! Various things in the standard library may implement one or more of the
//! three, where appropriate.
//!
//! # Implementing Iterator
//!
//! Creating an iterator of your own involves two steps: creating a `struct` to
//! hold the iterator's state, and then `impl`ementing [`Iterator`] for that
//! `struct`. This is why there are so many `struct`s in this module: there is
//! one for each iterator and iterator adapter.
//!
//! Let's make an iterator named `Counter` which counts from `1` to `5`:
//!
//! ```
//! // First, the struct:
//!
//! /// An iterator which counts from one to five
//! struct Counter {
//!     count: i32,
//! }
//!
//! // we want our count to start at one, so let's add a new() method to help.
//! // This isn't strictly necessary, but is convenient. Note that we start
//! // `count` at zero, we'll see why in `next()`'s implementation below.
//! impl Counter {
//!     fn new() -> Counter {
//!         Counter { count: 0 }
//!     }
//! }
//!
//! // Then, we implement `Iterator` for our `Counter`:
//!
//! impl Iterator for Counter {
//!     // we will be counting with i32
//!     type Item = i32;
//!
//!     // next() is the only required method
//!     fn next(&mut self) -> Option<i32> {
//!         // increment our count. This is why we started at zero.
//!         self.count += 1;
//!
//!         // check to see if we've finished counting or not.
//!         if self.count < 6 {
//!             Some(self.count)
//!         } else {
//!             None
//!         }
//!     }
//! }
//!
//! // And now we can use it!
//!
//! let mut counter = Counter::new();
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//! ```
//!
//! This will print `1` through `5`, each on their own line.
//!
//! Calling `next()` this way gets repetitive. Rust has a construct which can
//! call `next()` on your iterator, until it reaches `None`. Let's go over that
//! next.
//!
//! # for Loops and IntoIterator
//!
//! Rust's `for` loop syntax is actually sugar for iterators. Here's a basic
//! example of `for`:
//!
//! ```
//! let values = vec![1, 2, 3, 4, 5];
//!
//! for x in values {
//!     println!("{}", x);
//! }
//! ```
//!
//! This will print the numbers one through five, each on their own line. But
//! you'll notice something here: we never called anything on our vector to
//! produce an iterator. What gives?
//!
//! There's a trait in the standard library for converting something into an
//! iterator: [`IntoIterator`]. This trait has one method, [`into_iter()`],
//! which converts the thing implementing [`IntoIterator`] into an iterator.
//! Let's take a look at that `for` loop again, and what the compiler converts
//! it into:
//!
//! [`IntoIterator`]: trait.IntoIterator.html
//! [`into_iter()`]: trait.IntoIterator.html#tymethod.into_iter
//!
//! ```
//! let values = vec![1, 2, 3, 4, 5];
//!
//! for x in values {
//!     println!("{}", x);
//! }
//! ```
//!
//! Rust de-sugars this into:
//!
//! ```
//! let values = vec![1, 2, 3, 4, 5];
//! {
//!     let result = match values.into_iter() {
//!         mut iter => loop {
//!             match iter.next() {
//!                 Some(x) => { println!("{}", x); },
//!                 None => break,
//!             }
//!         },
//!     };
//!     result
//! }
//! ```
//!
//! First, we call `into_iter()` on the value. Then, we match on the iterator
//! that returns, calling [`next()`] over and over until we see a `None`. At
//! that point, we `break` out of the loop, and we're done iterating.
//!
//! There's one more subtle bit here: the standard library contains an
//! interesting implementation of [`IntoIterator`]:
//!
//! ```ignore
//! impl<I> IntoIterator for I where I: Iterator
//! ```
//!
//! In other words, all [`Iterator`]s implement [`IntoIterator`], by just
//! returning themselves. This means two things:
//!
//! 1. If you're writing an [`Iterator`], you can use it with a `for` loop.
//! 2. If you're creating a collection, implementing [`IntoIterator`] for it
//!    will allow your collection to be used with the `for` loop.
//!
//! # Adapters
//!
//! Functions which take an [`Iterator`] and return another [`Iterator`] are
//! often called 'iterator adapters', as they're a form of the 'adapter
//! pattern'.
//!
//! Common iterator adapters include [`map()`], [`take()`], and [`collect()`].
//! For more, see their documentation.
//!
//! [`map()`]: trait.Iterator.html#method.map
//! [`take()`]: trait.Iterator.html#method.take
//! [`collect()`]: trait.Iterator.html#method.collect
//!
//! # Laziness
//!
//! Iterators (and iterator [adapters](#adapters)) are *lazy*. This means that
//! just creating an iterator doesn't _do_ a whole lot. Nothing really happens
//! until you call [`next()`]. This is sometimes a source of confusion when
//! creating an iterator solely for its side effects. For example, the [`map()`]
//! method calls a closure on each element it iterates over:
//!
//! ```
//! let v = vec![1, 2, 3, 4, 5];
//! v.iter().map(|x| println!("{}", x));
//! ```
//!
//! This will not print any values, as we only created an iterator, rather than
//! using it. The compiler will warn us about this kind of behavior:
//!
//! ```text
//! warning: unused result which must be used: iterator adaptors are lazy and
//! do nothing unless consumed
//! ```
//!
//! The idiomatic way to write a [`map()`] for its side effects is to use a
//! `for` loop instead:
//!
//! ```
//! let v = vec![1, 2, 3, 4, 5];
//!
//! for x in &v {
//!     println!("{}", x);
//! }
//! ```
//!
//! [`map()`]: trait.Iterator.html#method.map
//!
//! The two most common ways to evaluate an iterator are to use a `for` loop
//! like this, or using the [`collect()`] adapter to produce a new collection.
//!
//! [`collect()`]: trait.Iterator.html#method.collect
//!
//! # Infinity
//!
//! Iterators do not have to be finite. As an example, an open-ended range is
//! an infinite iterator:
//!
//! ```
//! let numbers = 0..;
//! ```
//!
//! It is common to use the [`take()`] iterator adapter to turn an infinite
//! iterator into a finite one:
//!
//! ```
//! let numbers = 0..;
//! let five_numbers = numbers.take(5);
//!
//! for number in five_numbers {
//!     println!("{}", number);
//! }
//! ```
//!
//! This will print the numbers `0` through `4`, each on their own line.
//!
//! [`take()`]: trait.Iterator.html#method.take

#![stable(feature = "rust1", since = "1.0.0")]

use clone::Clone;
use cmp;
use cmp::{Ord, PartialOrd, PartialEq, Ordering};
use default::Default;
use marker;
use mem;
use num::{Zero, One};
use ops::{self, Add, Sub, FnMut, Mul, RangeFrom};
use option::Option::{self, Some, None};
use marker::Sized;
use usize;

fn _assert_is_object_safe(_: &Iterator<Item=()>) {}

/// An interface for dealing with "external iterators". These types of iterators
/// can be resumed at any time as all state is stored internally as opposed to
/// being located on the call stack.
///
/// The Iterator protocol states that an iterator yields a (potentially-empty,
/// potentially-infinite) sequence of values, and returns `None` to signal that
/// it's finished. The Iterator protocol does not define behavior after `None`
/// is returned. A concrete Iterator implementation may choose to behave however
/// it wishes, either by returning `None` infinitely, or by doing something
/// else.
#[lang = "iterator"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "`{Self}` is not an iterator; maybe try calling \
                            `.iter()` or a similar method"]
pub trait Iterator {
    /// The type of the elements being iterated
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Advances the iterator and returns the next value. Returns `None` when the
    /// end is reached.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next(&mut self) -> Option<Self::Item>;

    /// Returns a lower and upper bound on the remaining length of the iterator.
    ///
    /// An upper bound of `None` means either there is no known upper bound, or
    /// the upper bound does not fit within a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let it = (0..10).filter(|x| x % 2 == 0).chain(15..20);
    /// assert_eq!((5, Some(15)), it.size_hint());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn size_hint(&self) -> (usize, Option<usize>) { (0, None) }

    /// Counts the number of elements in this iterator.
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so counting elements of
    /// an iterator with more than `usize::MAX` elements either produces the
    /// wrong result or panics. If debug assertions are enabled, a panic is
    /// guaranteed.
    ///
    /// # Panics
    ///
    /// This functions might panic if the iterator has more than `usize::MAX`
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().count(), 5);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn count(self) -> usize where Self: Sized {
        // Might overflow.
        self.fold(0, |cnt, _| cnt + 1)
    }

    /// Loops through the entire iterator, returning the last element.
    ///
    /// # Examples
    ///
    /// ```
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

    /// Skips the `n` first elements of the iterator and returns the next one.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.nth(2), Some(&3));
    /// assert_eq!(it.nth(2), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> where Self: Sized {
        for x in self {
            if n == 0 { return Some(x) }
            n -= 1;
        }
        None
    }

    /// Chain this iterator with another, returning a new iterator that will
    /// finish iterating over the current iterator, and then iterate
    /// over the other specified iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [0];
    /// let b = [1];
    /// let mut it = a.iter().chain(&b);
    /// assert_eq!(it.next(), Some(&0));
    /// assert_eq!(it.next(), Some(&1));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chain<U>(self, other: U) -> Chain<Self, U::IntoIter> where
        Self: Sized, U: IntoIterator<Item=Self::Item>,
    {
        Chain{a: self, b: other.into_iter(), state: ChainState::Both}
    }

    /// Creates an iterator that iterates over both this and the specified
    /// iterators simultaneously, yielding the two elements as pairs. When
    /// either iterator returns `None`, all further invocations of `next()`
    /// will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [0];
    /// let b = [1];
    /// let mut it = a.iter().zip(&b);
    /// assert_eq!(it.next(), Some((&0, &1)));
    /// assert!(it.next().is_none());
    /// ```
    ///
    /// `zip` can provide similar functionality to `enumerate`:
    ///
    /// ```
    /// for pair in "foo".chars().enumerate() {
    ///     println!("{:?}", pair);
    /// }
    ///
    /// for pair in (0..).zip("foo".chars()) {
    ///     println!("{:?}", pair);
    /// }
    /// ```
    ///
    /// both produce the same output.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn zip<U>(self, other: U) -> Zip<Self, U::IntoIter> where
        Self: Sized, U: IntoIterator
    {
        Zip{a: self, b: other.into_iter()}
    }

    /// Creates a new iterator that will apply the specified function to each
    /// element returned by the first, yielding the mapped element instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2];
    /// let mut it = a.iter().map(|&x| 2 * x);
    /// assert_eq!(it.next(), Some(2));
    /// assert_eq!(it.next(), Some(4));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn map<B, F>(self, f: F) -> Map<Self, F> where
        Self: Sized, F: FnMut(Self::Item) -> B,
    {
        Map{iter: self, f: f}
    }

    /// Creates an iterator that applies the predicate to each element returned
    /// by this iterator. The only elements that will be yielded are those that
    /// make the predicate evaluate to `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2];
    /// let mut it = a.iter().filter(|&x| *x > 1);
    /// assert_eq!(it.next(), Some(&2));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter<P>(self, predicate: P) -> Filter<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
    {
        Filter{iter: self, predicate: predicate}
    }

    /// Creates an iterator that both filters and maps elements.
    /// If the specified function returns `None`, the element is skipped.
    /// Otherwise the option is unwrapped and the new value is yielded.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2];
    /// let mut it = a.iter().filter_map(|&x| if x > 1 {Some(2 * x)} else {None});
    /// assert_eq!(it.next(), Some(4));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F> where
        Self: Sized, F: FnMut(Self::Item) -> Option<B>,
    {
        FilterMap { iter: self, f: f }
    }

    /// Creates an iterator that yields pairs `(i, val)` where `i` is the
    /// current index of iteration and `val` is the value returned by the
    /// iterator.
    ///
    /// `enumerate` keeps its count as a `usize`. If you want to count by a
    /// different sized integer, the `zip` function provides similar
    /// functionality.
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so enumerating more than
    /// `usize::MAX` elements either produces the wrong result or panics. If
    /// debug assertions are enabled, a panic is guaranteed.
    ///
    /// # Panics
    ///
    /// The returned iterator might panic if the to-be-returned index would
    /// overflow a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [100, 200];
    /// let mut it = a.iter().enumerate();
    /// assert_eq!(it.next(), Some((0, &100)));
    /// assert_eq!(it.next(), Some((1, &200)));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn enumerate(self) -> Enumerate<Self> where Self: Sized {
        Enumerate { iter: self, count: 0 }
    }

    /// Creates an iterator that has a `.peek()` method
    /// that returns an optional reference to the next element.
    ///
    /// # Examples
    ///
    /// ```
    /// let xs = [100, 200, 300];
    /// let mut it = xs.iter().cloned().peekable();
    /// assert_eq!(*it.peek().unwrap(), 100);
    /// assert_eq!(it.next().unwrap(), 100);
    /// assert_eq!(it.next().unwrap(), 200);
    /// assert_eq!(*it.peek().unwrap(), 300);
    /// assert_eq!(*it.peek().unwrap(), 300);
    /// assert_eq!(it.next().unwrap(), 300);
    /// assert!(it.peek().is_none());
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn peekable(self) -> Peekable<Self> where Self: Sized {
        Peekable{iter: self, peeked: None}
    }

    /// Creates an iterator that invokes the predicate on elements
    /// until it returns false. Once the predicate returns false, that
    /// element and all further elements are yielded.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().skip_while(|&a| *a < 3);
    /// assert_eq!(it.next(), Some(&3));
    /// assert_eq!(it.next(), Some(&4));
    /// assert_eq!(it.next(), Some(&5));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
    {
        SkipWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator that yields elements so long as the predicate
    /// returns true. After the predicate returns false for the first time, no
    /// further elements will be yielded.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().take_while(|&a| *a < 3);
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P> where
        Self: Sized, P: FnMut(&Self::Item) -> bool,
    {
        TakeWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator that skips the first `n` elements of this iterator,
    /// and then yields all further items.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().skip(3);
    /// assert_eq!(it.next(), Some(&4));
    /// assert_eq!(it.next(), Some(&5));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn skip(self, n: usize) -> Skip<Self> where Self: Sized {
        Skip{iter: self, n: n}
    }

    /// Creates an iterator that yields the first `n` elements of this
    /// iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().take(3);
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert_eq!(it.next(), Some(&3));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn take(self, n: usize) -> Take<Self> where Self: Sized, {
        Take{iter: self, n: n}
    }

    /// Creates a new iterator that behaves in a similar fashion to fold.
    /// There is a state which is passed between each iteration and can be
    /// mutated as necessary. The yielded values from the closure are yielded
    /// from the Scan instance when not `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().scan(1, |fac, &x| {
    ///   *fac = *fac * x;
    ///   Some(*fac)
    /// });
    /// assert_eq!(it.next(), Some(1));
    /// assert_eq!(it.next(), Some(2));
    /// assert_eq!(it.next(), Some(6));
    /// assert_eq!(it.next(), Some(24));
    /// assert_eq!(it.next(), Some(120));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
        where Self: Sized, F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        Scan{iter: self, f: f, state: initial_state}
    }

    /// Takes a function that maps each element to a new iterator and yields
    /// all the elements of the produced iterators.
    ///
    /// This is useful for unraveling nested structures.
    ///
    /// # Examples
    ///
    /// ```
    /// let words = ["alpha", "beta", "gamma"];
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

    /// Creates an iterator that yields `None` forever after the underlying
    /// iterator yields `None`. Random-access iterator behavior is not
    /// affected, only single and double-ended iterator behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// fn process<U: Iterator<Item=i32>>(it: U) -> i32 {
    ///     let mut it = it.fuse();
    ///     let mut sum = 0;
    ///     for x in it.by_ref() {
    ///         if x > 5 {
    ///             break;
    ///         }
    ///         sum += x;
    ///     }
    ///     // did we exhaust the iterator?
    ///     if it.next().is_none() {
    ///         sum += 1000;
    ///     }
    ///     sum
    /// }
    /// let x = vec![1, 2, 3, 7, 8, 9];
    /// assert_eq!(process(x.into_iter()), 6);
    /// let x = vec![1, 2, 3];
    /// assert_eq!(process(x.into_iter()), 1006);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn fuse(self) -> Fuse<Self> where Self: Sized {
        Fuse{iter: self, done: false}
    }

    /// Creates an iterator that calls a function with a reference to each
    /// element before yielding it. This is often useful for debugging an
    /// iterator pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 4, 2, 3, 8, 9, 6];
    /// let sum: i32 = a.iter()
    ///                 .map(|x| *x)
    ///                 .inspect(|&x| println!("filtering {}", x))
    ///                 .filter(|&x| x % 2 == 0)
    ///                 .inspect(|&x| println!("{} made it through", x))
    ///                 .fold(0, |sum, i| sum + i);
    /// println!("{}", sum);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn inspect<F>(self, f: F) -> Inspect<Self, F> where
        Self: Sized, F: FnMut(&Self::Item),
    {
        Inspect{iter: self, f: f}
    }

    /// Creates a wrapper around a mutable reference to the iterator.
    ///
    /// This is useful to allow applying iterator adaptors while still
    /// retaining ownership of the original iterator value.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut it = 0..10;
    /// // sum the first five values
    /// let partial_sum = it.by_ref().take(5).fold(0, |a, b| a + b);
    /// assert_eq!(partial_sum, 10);
    /// assert_eq!(it.next(), Some(5));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn by_ref(&mut self) -> &mut Self where Self: Sized { self }

    /// Loops through the entire iterator, collecting all of the elements into
    /// a container implementing `FromIterator`.
    ///
    /// # Examples
    ///
    /// ```
    /// let expected = [1, 2, 3, 4, 5];
    /// let actual: Vec<_> = expected.iter().cloned().collect();
    /// assert_eq!(actual, expected);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn collect<B: FromIterator<Self::Item>>(self) -> B where Self: Sized {
        FromIterator::from_iter(self)
    }

    /// Loops through the entire iterator, collecting all of the elements into
    /// one of two containers, depending on a predicate. The elements of the
    /// first container satisfy the predicate, while the elements of the second
    /// do not.
    ///
    /// ```
    /// let vec = vec![1, 2, 3, 4];
    /// let (even, odd): (Vec<_>, Vec<_>) = vec.into_iter().partition(|&n| n % 2 == 0);
    /// assert_eq!(even, [2, 4]);
    /// assert_eq!(odd, [1, 3]);
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

    /// Performs a fold operation over the entire iterator, returning the
    /// eventual state at the end of the iteration.
    ///
    /// This operation is sometimes called 'reduce' or 'inject'.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().fold(0, |acc, &item| acc + item), 15);
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

    /// Tests whether the predicate holds true for all elements in the iterator.
    ///
    /// Does not consume the iterator past the first non-matching element.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().all(|x| *x > 0));
    /// assert!(!a.iter().all(|x| *x > 2));
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

    /// Tests whether any element of an iterator satisfies the specified
    /// predicate.
    ///
    /// Does not consume the iterator past the first found element.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.any(|x| *x == 3));
    /// assert_eq!(it.collect::<Vec<_>>(), [&4, &5]);
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

    /// Returns the first element satisfying the specified predicate.
    ///
    /// Does not consume the iterator past the first found element.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.find(|&x| *x == 3), Some(&3));
    /// assert_eq!(it.collect::<Vec<_>>(), [&4, &5]);
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

    /// Returns the index of the first element satisfying the specified predicate
    ///
    /// Does not consume the iterator past the first found element.
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so if there are more
    /// than `usize::MAX` non-matching elements, it either produces the wrong
    /// result or panics. If debug assertions are enabled, a panic is
    /// guaranteed.
    ///
    /// # Panics
    ///
    /// This functions might panic if the iterator has more than `usize::MAX`
    /// non-matching elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.position(|x| *x == 3), Some(2));
    /// assert_eq!(it.collect::<Vec<_>>(), [&4, &5]);
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

    /// Returns the index of the last element satisfying the specified predicate
    ///
    /// If no element matches, `None` is returned.
    ///
    /// Does not consume the iterator *before* the first found element.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 2, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.rposition(|x| *x == 2), Some(2));
    /// assert_eq!(it.collect::<Vec<_>>(), [&1, &2]);
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
        P: FnMut(Self::Item) -> bool,
        Self: Sized + ExactSizeIterator + DoubleEndedIterator
    {
        let mut i = self.len();

        while let Some(v) = self.next_back() {
            if predicate(v) {
                return Some(i - 1);
            }
            // No need for an overflow check here, because `ExactSizeIterator`
            // implies that the number of elements fits into a `usize`.
            i -= 1;
        }
        None
    }

    /// Consumes the entire iterator to return the maximum element.
    ///
    /// Returns the rightmost element if the comparison determines two elements
    /// to be equally maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().max(), Some(&5));
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

    /// Consumes the entire iterator to return the minimum element.
    ///
    /// Returns the leftmost element if the comparison determines two elements
    /// to be equally minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert_eq!(a.iter().min(), Some(&1));
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
    /// Returns the rightmost element if the comparison determines two elements
    /// to be equally maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_cmp)]
    ///
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().max_by(|x| x.abs()).unwrap(), -10);
    /// ```
    #[inline]
    #[unstable(feature = "iter_cmp",
               reason = "may want to produce an Ordering directly; see #15311",
               issue = "27724")]
    fn max_by<B: Ord, F>(self, f: F) -> Option<Self::Item> where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        select_fold1(self,
                     f,
                     // switch to y even if it is only equal, to preserve
                     // stability.
                     |x_p, _, y_p, _| x_p <= y_p)
            .map(|(_, x)| x)
    }

    /// Returns the element that gives the minimum value from the
    /// specified function.
    ///
    /// Returns the leftmost element if the comparison determines two elements
    /// to be equally minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_cmp)]
    ///
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().min_by(|x| x.abs()).unwrap(), 0);
    /// ```
    #[inline]
    #[unstable(feature = "iter_cmp",
               reason = "may want to produce an Ordering directly; see #15311",
               issue = "27724")]
    fn min_by<B: Ord, F>(self, f: F) -> Option<Self::Item> where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        select_fold1(self,
                     f,
                     // only switch to y if it is strictly smaller, to
                     // preserve stability.
                     |x_p, _, y_p, _| x_p > y_p)
            .map(|(_, x)| x)
    }

    /// Change the direction of the iterator
    ///
    /// The flipped iterator swaps the ends on an iterator that can already
    /// be iterated from the front and from the back.
    ///
    ///
    /// If the iterator also implements RandomAccessIterator, the flipped
    /// iterator is also random access, with the indices starting at the back
    /// of the original iterator.
    ///
    /// Note: Random access with flipped indices still only applies to the first
    /// `std::usize::MAX` elements of the original iterator.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rev(self) -> Rev<Self> where Self: Sized + DoubleEndedIterator {
        Rev{iter: self}
    }

    /// Converts an iterator of pairs into a pair of containers.
    ///
    /// Loops through the entire iterator, collecting the first component of
    /// each item into one new container, and the second component into another.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [(1, 2), (3, 4)];
    /// let (left, right): (Vec<_>, Vec<_>) = a.iter().cloned().unzip();
    /// assert_eq!(left, [1, 3]);
    /// assert_eq!(right, [2, 4]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB) where
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
        Self: Sized + Iterator<Item=(A, B)>,
    {
        struct SizeHint<A>(usize, Option<usize>, marker::PhantomData<A>);
        impl<A> Iterator for SizeHint<A> {
            type Item = A;

            fn next(&mut self) -> Option<A> { None }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.0, self.1)
            }
        }

        let (lo, hi) = self.size_hint();
        let mut ts: FromA = Default::default();
        let mut us: FromB = Default::default();

        ts.extend(SizeHint(lo, hi, marker::PhantomData));
        us.extend(SizeHint(lo, hi, marker::PhantomData));

        for (t, u) in self {
            ts.extend(Some(t));
            us.extend(Some(u));
        }

        (ts, us)
    }

    /// Creates an iterator that clones the elements it yields.
    ///
    /// This is useful for converting an `Iterator<&T>` to an`Iterator<T>`,
    /// so it's a more convenient form of `map(|&x| x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [0, 1, 2];
    /// let v_cloned: Vec<_> = a.iter().cloned().collect();
    /// let v_map: Vec<_> = a.iter().map(|&x| x).collect();
    /// assert_eq!(v_cloned, v_map);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn cloned<'a, T: 'a>(self) -> Cloned<Self>
        where Self: Sized + Iterator<Item=&'a T>, T: Clone
    {
        Cloned { it: self }
    }

    /// Repeats an iterator endlessly
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2];
    /// let mut it = a.iter().cycle();
    /// assert_eq!(it.next(), Some(&1));
    /// assert_eq!(it.next(), Some(&2));
    /// assert_eq!(it.next(), Some(&1));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn cycle(self) -> Cycle<Self> where Self: Sized + Clone {
        Cycle{orig: self.clone(), iter: self}
    }

    /// Iterates over the entire iterator, summing up all the elements
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_arith)]
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// let it = a.iter();
    /// assert_eq!(it.sum::<i32>(), 15);
    /// ```
    #[unstable(feature = "iter_arith", reason = "bounds recently changed",
               issue = "27739")]
    fn sum<S=<Self as Iterator>::Item>(self) -> S where
        S: Add<Self::Item, Output=S> + Zero,
        Self: Sized,
    {
        self.fold(Zero::zero(), |s, e| s + e)
    }

    /// Iterates over the entire iterator, multiplying all the elements
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(iter_arith)]
    ///
    /// fn factorial(n: u32) -> u32 {
    ///     (1..).take_while(|&i| i <= n).product()
    /// }
    /// assert_eq!(factorial(0), 1);
    /// assert_eq!(factorial(1), 1);
    /// assert_eq!(factorial(5), 120);
    /// ```
    #[unstable(feature="iter_arith", reason = "bounds recently changed",
               issue = "27739")]
    fn product<P=<Self as Iterator>::Item>(self) -> P where
        P: Mul<Self::Item, Output=P> + One,
        Self: Sized,
    {
        self.fold(One::one(), |p, e| p * e)
    }

    /// Lexicographically compares the elements of this `Iterator` with those
    /// of another.
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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
    #[unstable(feature = "iter_order", reason = "needs review and revision", issue = "27737")]
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

/// Select an element from an iterator based on the given projection
/// and "comparison" function.
///
/// This is an idiosyncratic helper to try to factor out the
/// commonalities of {max,min}{,_by}. In particular, this avoids
/// having to implement optimizations several times.
#[inline]
fn select_fold1<I,B, FProj, FCmp>(mut it: I,
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
            if f_cmp(&sel_p,  &sel, &x_p, &x) {
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
}

/// Conversion from an `Iterator`
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented="a collection of type `{Self}` cannot be \
                          built from an iterator over elements of type `{A}`"]
pub trait FromIterator<A> {
    /// Builds a container with elements from something iterable.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use std::iter::FromIterator;
    ///
    /// let colors_vec = vec!["red", "red", "yellow", "blue"];
    /// let colors_set = HashSet::<&str>::from_iter(colors_vec);
    /// assert_eq!(colors_set.len(), 3);
    /// ```
    ///
    /// `FromIterator` is more commonly used implicitly via the
    /// `Iterator::collect` method:
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let colors_vec = vec!["red", "red", "yellow", "blue"];
    /// let colors_set = colors_vec.into_iter().collect::<HashSet<&str>>();
    /// assert_eq!(colors_set.len(), 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_iter<T: IntoIterator<Item=A>>(iterator: T) -> Self;
}

/// Conversion into an `Iterator`
///
/// Implementing this trait allows you to use your type with Rust's `for` loop. See
/// the [module level documentation](index.html) for more details.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IntoIterator {
    /// The type of the elements being iterated
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// A container for iterating over elements of type `Item`
    #[stable(feature = "rust1", since = "1.0.0")]
    type IntoIter: Iterator<Item=Self::Item>;

    /// Consumes `Self` and returns an iterator over it
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_iter(self) -> Self::IntoIter;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> IntoIterator for I {
    type Item = I::Item;
    type IntoIter = I;

    fn into_iter(self) -> I {
        self
    }
}

/// A type growable from an `Iterator` implementation
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Extend<A> {
    /// Extends a container with the elements yielded by an arbitrary iterator
    #[stable(feature = "rust1", since = "1.0.0")]
    fn extend<T: IntoIterator<Item=A>>(&mut self, iterable: T);
}

/// A range iterator able to yield elements from both ends
///
/// A `DoubleEndedIterator` can be thought of as a deque in that `next()` and
/// `next_back()` exhaust elements from the *same* range, and do not work
/// independently of each other.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait DoubleEndedIterator: Iterator {
    /// Yields an element from the end of the range, returning `None` if the
    /// range is empty.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next_back(&mut self) -> Option<Self::Item>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for &'a mut I {
    fn next_back(&mut self) -> Option<I::Item> { (**self).next_back() }
}

/// An iterator that knows its exact length
///
/// This trait is a helper for iterators like the vector iterator, so that
/// it can support double-ended enumeration.
///
/// `Iterator::size_hint` *must* return the exact size of the iterator.
/// Note that the size must fit in `usize`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExactSizeIterator: Iterator {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Returns the exact length of the iterator.
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: ExactSizeIterator + ?Sized> ExactSizeIterator for &'a mut I {}

// All adaptors that preserve the size of the wrapped iterator are fine
// Adaptors that may overflow in `size_hint` are not, i.e. `Chain`.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Enumerate<I> where I: ExactSizeIterator {}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator, F> ExactSizeIterator for Inspect<I, F> where
    F: FnMut(&I::Item),
{}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Rev<I>
    where I: ExactSizeIterator + DoubleEndedIterator {}
#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: ExactSizeIterator, F> ExactSizeIterator for Map<I, F> where
    F: FnMut(I::Item) -> B,
{}
#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> ExactSizeIterator for Zip<A, B>
    where A: ExactSizeIterator, B: ExactSizeIterator {}

/// An double-ended iterator with the direction inverted
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rev<T> {
    iter: T
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Rev<I> where I: DoubleEndedIterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> { self.iter.next_back() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Rev<I> where I: DoubleEndedIterator {
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> { self.iter.next() }
}

/// An iterator that clones the elements of an underlying iterator
#[stable(feature = "iter_cloned", since = "1.1.0")]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct Cloned<I> {
    it: I,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I, T: 'a> Iterator for Cloned<I>
    where I: Iterator<Item=&'a T>, T: Clone
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.it.next().cloned()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I, T: 'a> DoubleEndedIterator for Cloned<I>
    where I: DoubleEndedIterator<Item=&'a T>, T: Clone
{
    fn next_back(&mut self) -> Option<T> {
        self.it.next_back().cloned()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I, T: 'a> ExactSizeIterator for Cloned<I>
    where I: ExactSizeIterator<Item=&'a T>, T: Clone
{}

/// An iterator that repeats endlessly
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Cycle<I> {
    orig: I,
    iter: I,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Cycle<I> where I: Clone + Iterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        match self.iter.next() {
            None => { self.iter = self.orig.clone(); self.iter.next() }
            y => y
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // the cycle iterator is either empty or infinite
        match self.orig.size_hint() {
            sz @ (0, Some(0)) => sz,
            (0, _) => (0, None),
            _ => (usize::MAX, None)
        }
    }
}

/// An iterator that strings two iterators together
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chain<A, B> {
    a: A,
    b: B,
    state: ChainState,
}

// The iterator protocol specifies that iteration ends with the return value
// `None` from `.next()` (or `.next_back()`) and it is unspecified what
// further calls return. The chain adaptor must account for this since it uses
// two subiterators.
//
//  It uses three states:
//
//  - Both: `a` and `b` are remaining
//  - Front: `a` remaining
//  - Back: `b` remaining
//
//  The fourth state (neither iterator is remaining) only occurs after Chain has
//  returned None once, so we don't need to store this state.
#[derive(Clone)]
enum ChainState {
    // both front and back iterator are remaining
    Both,
    // only front is remaining
    Front,
    // only back is remaining
    Back,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Chain<A, B> where
    A: Iterator,
    B: Iterator<Item = A::Item>
{
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        match self.state {
            ChainState::Both => match self.a.next() {
                elt @ Some(..) => elt,
                None => {
                    self.state = ChainState::Back;
                    self.b.next()
                }
            },
            ChainState::Front => self.a.next(),
            ChainState::Back => self.b.next(),
        }
    }

    #[inline]
    fn count(self) -> usize {
        match self.state {
            ChainState::Both => self.a.count() + self.b.count(),
            ChainState::Front => self.a.count(),
            ChainState::Back => self.b.count(),
        }
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<A::Item> {
        match self.state {
            ChainState::Both | ChainState::Front => {
                for x in self.a.by_ref() {
                    if n == 0 {
                        return Some(x)
                    }
                    n -= 1;
                }
                if let ChainState::Both = self.state {
                    self.state = ChainState::Back;
                }
            }
            ChainState::Back => {}
        }
        if let ChainState::Back = self.state {
            self.b.nth(n)
        } else {
            None
        }
    }

    #[inline]
    fn last(self) -> Option<A::Item> {
        match self.state {
            ChainState::Both => {
                // Must exhaust a before b.
                let a_last = self.a.last();
                let b_last = self.b.last();
                b_last.or(a_last)
            },
            ChainState::Front => self.a.last(),
            ChainState::Back => self.b.last()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = a_lower.saturating_add(b_lower);

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) => x.checked_add(y),
            _ => None
        };

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Chain<A, B> where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item=A::Item>,
{
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        match self.state {
            ChainState::Both => match self.b.next_back() {
                elt @ Some(..) => elt,
                None => {
                    self.state = ChainState::Front;
                    self.a.next_back()
                }
            },
            ChainState::Front => self.a.next_back(),
            ChainState::Back => self.b.next_back(),
        }
    }
}

/// An iterator that iterates two other iterators simultaneously
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Zip<A, B> {
    a: A,
    b: B
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Zip<A, B> where A: Iterator, B: Iterator
{
    type Item = (A::Item, B::Item);

    #[inline]
    fn next(&mut self) -> Option<(A::Item, B::Item)> {
        self.a.next().and_then(|x| {
            self.b.next().and_then(|y| {
                Some((x, y))
            })
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = cmp::min(a_lower, b_lower);

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) => Some(cmp::min(x,y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None
        };

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Zip<A, B> where
    A: DoubleEndedIterator + ExactSizeIterator,
    B: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(A::Item, B::Item)> {
        let a_sz = self.a.len();
        let b_sz = self.b.len();
        if a_sz != b_sz {
            // Adjust a, b to equal length
            if a_sz > b_sz {
                for _ in 0..a_sz - b_sz { self.a.next_back(); }
            } else {
                for _ in 0..b_sz - a_sz { self.b.next_back(); }
            }
        }
        match (self.a.next_back(), self.b.next_back()) {
            (Some(x), Some(y)) => Some((x, y)),
            (None, None) => None,
            _ => unreachable!(),
        }
    }
}

/// An iterator that maps the values of `iter` with `f`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Map<I, F> {
    iter: I,
    f: F,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: Iterator, F> Iterator for Map<I, F> where F: FnMut(I::Item) -> B {
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().map(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: DoubleEndedIterator, F> DoubleEndedIterator for Map<I, F> where
    F: FnMut(I::Item) -> B,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        self.iter.next_back().map(&mut self.f)
    }
}

/// An iterator that filters the elements of `iter` with `predicate`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Filter<I, P> {
    iter: I,
    predicate: P,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for Filter<I, P> where P: FnMut(&I::Item) -> bool {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        for x in self.iter.by_ref() {
            if (self.predicate)(&x) {
                return Some(x);
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, P> DoubleEndedIterator for Filter<I, P>
    where P: FnMut(&I::Item) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        for x in self.iter.by_ref().rev() {
            if (self.predicate)(&x) {
                return Some(x);
            }
        }
        None
    }
}

/// An iterator that uses `f` to both filter and map elements from `iter`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct FilterMap<I, F> {
    iter: I,
    f: F,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: Iterator, F> Iterator for FilterMap<I, F>
    where F: FnMut(I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        for x in self.iter.by_ref() {
            if let Some(y) = (self.f)(x) {
                return Some(y);
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: DoubleEndedIterator, F> DoubleEndedIterator for FilterMap<I, F>
    where F: FnMut(I::Item) -> Option<B>,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        for x in self.iter.by_ref().rev() {
            if let Some(y) = (self.f)(x) {
                return Some(y);
            }
        }
        None
    }
}

/// An iterator that yields the current count and the element during iteration
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Enumerate<I> {
    iter: I,
    count: usize,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Enumerate<I> where I: Iterator {
    type Item = (usize, <I as Iterator>::Item);

    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so enumerating more than
    /// `usize::MAX` elements either produces the wrong result or panics. If
    /// debug assertions are enabled, a panic is guaranteed.
    ///
    /// # Panics
    ///
    /// Might panic if the index of the element overflows a `usize`.
    #[inline]
    fn next(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        self.iter.next().map(|a| {
            let ret = (self.count, a);
            // Possible undefined overflow.
            self.count += 1;
            ret
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<(usize, I::Item)> {
        self.iter.nth(n).map(|a| {
            let i = self.count + n;
            self.count = i + 1;
            (i, a)
        })
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Enumerate<I> where
    I: ExactSizeIterator + DoubleEndedIterator
{
    #[inline]
    fn next_back(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        self.iter.next_back().map(|a| {
            let len = self.iter.len();
            // Can safely add, `ExactSizeIterator` promises that the number of
            // elements fits into a `usize`.
            (self.count + len, a)
        })
    }
}

/// An iterator with a `peek()` that returns an optional reference to the next element.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Peekable<I: Iterator> {
    iter: I,
    peeked: Option<I::Item>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        match self.peeked {
            Some(_) => self.peeked.take(),
            None => self.iter.next(),
        }
    }

    #[inline]
    fn count(self) -> usize {
        (if self.peeked.is_some() { 1 } else { 0 }) + self.iter.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        match self.peeked {
            Some(_) if n == 0 => self.peeked.take(),
            Some(_) => {
                self.peeked = None;
                self.iter.nth(n-1)
            },
            None => self.iter.nth(n)
        }
    }

    #[inline]
    fn last(self) -> Option<I::Item> {
        self.iter.last().or(self.peeked)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        if self.peeked.is_some() {
            let lo = lo.saturating_add(1);
            let hi = hi.and_then(|x| x.checked_add(1));
            (lo, hi)
        } else {
            (lo, hi)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator> ExactSizeIterator for Peekable<I> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> Peekable<I> {
    /// Returns a reference to the next element of the iterator with out
    /// advancing it, or None if the iterator is exhausted.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn peek(&mut self) -> Option<&I::Item> {
        if self.peeked.is_none() {
            self.peeked = self.iter.next();
        }
        match self.peeked {
            Some(ref value) => Some(value),
            None => None,
        }
    }

    /// Checks whether peekable iterator is empty or not.
    #[inline]
    pub fn is_empty(&mut self) -> bool {
        self.peek().is_none()
    }
}

/// An iterator that rejects elements while `predicate` is true
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct SkipWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for SkipWhile<I, P>
    where P: FnMut(&I::Item) -> bool
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        for x in self.iter.by_ref() {
            if self.flag || !(self.predicate)(&x) {
                self.flag = true;
                return Some(x);
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

/// An iterator that only accepts elements while `predicate` is true
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct TakeWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for TakeWhile<I, P>
    where P: FnMut(&I::Item) -> bool
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.flag {
            None
        } else {
            self.iter.next().and_then(|x| {
                if (self.predicate)(&x) {
                    Some(x)
                } else {
                    self.flag = true;
                    None
                }
            })
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

/// An iterator that skips over `n` elements of `iter`.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Skip<I> {
    iter: I,
    n: usize
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Skip<I> where I: Iterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.n == 0 {
            self.iter.next()
        } else {
            let old_n = self.n;
            self.n = 0;
            self.iter.nth(old_n)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        // Can't just add n + self.n due to overflow.
        if self.n == 0 {
            self.iter.nth(n)
        } else {
            let to_skip = self.n;
            self.n = 0;
            // nth(n) skips n+1
            if self.iter.nth(to_skip-1).is_none() {
                return None;
            }
            self.iter.nth(n)
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count().saturating_sub(self.n)
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        if self.n == 0 {
            self.iter.last()
        } else {
            let next = self.next();
            if next.is_some() {
                // recurse. n should be 0.
                self.last().or(next)
            } else {
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = lower.saturating_sub(self.n);
        let upper = upper.map(|x| x.saturating_sub(self.n));

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Skip<I> where I: ExactSizeIterator {}

/// An iterator that only iterates over the first `n` iterations of `iter`.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Take<I> {
    iter: I,
    n: usize
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Take<I> where I: Iterator{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        if self.n != 0 {
            self.n -= 1;
            self.iter.next()
        } else {
            None
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        if self.n > n {
            self.n -= n + 1;
            self.iter.nth(n)
        } else {
            if self.n > 0 {
                self.iter.nth(self.n - 1);
                self.n = 0;
            }
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = cmp::min(lower, self.n);

        let upper = match upper {
            Some(x) if x < self.n => Some(x),
            _ => Some(self.n)
        };

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Take<I> where I: ExactSizeIterator {}


/// An iterator to maintain state while iterating another iterator
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Scan<I, St, F> {
    iter: I,
    f: F,
    state: St,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I, St, F> Iterator for Scan<I, St, F> where
    I: Iterator,
    F: FnMut(&mut St, I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().and_then(|a| (self.f)(&mut self.state, a))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the scan function
    }
}

/// An iterator that maps each element to an iterator,
/// and yields the elements of the produced iterators
///
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct FlatMap<I, U: IntoIterator, F> {
    iter: I,
    f: F,
    frontiter: Option<U::IntoIter>,
    backiter: Option<U::IntoIter>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, U: IntoIterator, F> Iterator for FlatMap<I, U, F>
    where F: FnMut(I::Item) -> U,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.frontiter {
                if let Some(x) = inner.by_ref().next() {
                    return Some(x)
                }
            }
            match self.iter.next().map(&mut self.f) {
                None => return self.backiter.as_mut().and_then(|it| it.next()),
                next => self.frontiter = next.map(IntoIterator::into_iter),
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (flo, fhi) = self.frontiter.as_ref().map_or((0, Some(0)), |it| it.size_hint());
        let (blo, bhi) = self.backiter.as_ref().map_or((0, Some(0)), |it| it.size_hint());
        let lo = flo.saturating_add(blo);
        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(b)),
            _ => (lo, None)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, U, F> DoubleEndedIterator for FlatMap<I, U, F> where
    F: FnMut(I::Item) -> U,
    U: IntoIterator,
    U::IntoIter: DoubleEndedIterator
{
    #[inline]
    fn next_back(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.backiter {
                if let Some(y) = inner.next_back() {
                    return Some(y)
                }
            }
            match self.iter.next_back().map(&mut self.f) {
                None => return self.frontiter.as_mut().and_then(|it| it.next_back()),
                next => self.backiter = next.map(IntoIterator::into_iter),
            }
        }
    }
}

/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
///
/// These can be created through
/// [`iter.fuse()`](trait.Iterator.html#method.fuse).
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Fuse<I> {
    iter: I,
    done: bool
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Fuse<I> where I: Iterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        if self.done {
            None
        } else {
            let next = self.iter.next();
            self.done = next.is_none();
            next
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        if self.done {
            None
        } else {
            let nth = self.iter.nth(n);
            self.done = nth.is_none();
            nth
        }
    }

    #[inline]
    fn last(self) -> Option<I::Item> {
        if self.done {
            None
        } else {
            self.iter.last()
        }
    }

    #[inline]
    fn count(self) -> usize {
        if self.done {
            0
        } else {
            self.iter.count()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            self.iter.size_hint()
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Fuse<I> where I: DoubleEndedIterator {
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        if self.done {
            None
        } else {
            let next = self.iter.next_back();
            self.done = next.is_none();
            next
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Fuse<I> where I: ExactSizeIterator {}

/// An iterator that calls a function with a reference to each
/// element before yielding it.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Inspect<I, F> {
    iter: I,
    f: F,
}

impl<I: Iterator, F> Inspect<I, F> where F: FnMut(&I::Item) {
    #[inline]
    fn do_inspect(&mut self, elt: Option<I::Item>) -> Option<I::Item> {
        if let Some(ref a) = elt {
            (self.f)(a);
        }

        elt
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, F> Iterator for Inspect<I, F> where F: FnMut(&I::Item) {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        let next = self.iter.next();
        self.do_inspect(next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, F> DoubleEndedIterator for Inspect<I, F>
    where F: FnMut(&I::Item),
{
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        let next = self.iter.next_back();
        self.do_inspect(next)
    }
}

/// Objects that can be stepped over in both directions.
///
/// The `steps_between` function provides a way to efficiently compare
/// two `Step` objects.
#[unstable(feature = "step_trait",
           reason = "likely to be replaced by finer-grained traits",
           issue = "27741")]
pub trait Step: PartialOrd + Sized {
    /// Steps `self` if possible.
    fn step(&self, by: &Self) -> Option<Self>;

    /// Returns the number of steps between two step objects. The count is
    /// inclusive of `start` and exclusive of `end`.
    ///
    /// Returns `None` if it is not possible to calculate `steps_between`
    /// without overflow.
    fn steps_between(start: &Self, end: &Self, by: &Self) -> Option<usize>;
}

macro_rules! step_impl_unsigned {
    ($($t:ty)*) => ($(
        impl Step for $t {
            #[inline]
            fn step(&self, by: &$t) -> Option<$t> {
                (*self).checked_add(*by)
            }
            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t, by: &$t) -> Option<usize> {
                if *by == 0 { return None; }
                if *start < *end {
                    // Note: We assume $t <= usize here
                    let diff = (*end - *start) as usize;
                    let by = *by as usize;
                    if diff % by > 0 {
                        Some(diff / by + 1)
                    } else {
                        Some(diff / by)
                    }
                } else {
                    Some(0)
                }
            }
        }
    )*)
}
macro_rules! step_impl_signed {
    ($($t:ty)*) => ($(
        impl Step for $t {
            #[inline]
            fn step(&self, by: &$t) -> Option<$t> {
                (*self).checked_add(*by)
            }
            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t, by: &$t) -> Option<usize> {
                if *by == 0 { return None; }
                let diff: usize;
                let by_u: usize;
                if *by > 0 {
                    if *start >= *end {
                        return Some(0);
                    }
                    // Note: We assume $t <= isize here
                    // Use .wrapping_sub and cast to usize to compute the
                    // difference that may not fit inside the range of isize.
                    diff = (*end as isize).wrapping_sub(*start as isize) as usize;
                    by_u = *by as usize;
                } else {
                    if *start <= *end {
                        return Some(0);
                    }
                    diff = (*start as isize).wrapping_sub(*end as isize) as usize;
                    by_u = (*by as isize).wrapping_mul(-1) as usize;
                }
                if diff % by_u > 0 {
                    Some(diff / by_u + 1)
                } else {
                    Some(diff / by_u)
                }
            }
        }
    )*)
}

macro_rules! step_impl_no_between {
    ($($t:ty)*) => ($(
        impl Step for $t {
            #[inline]
            fn step(&self, by: &$t) -> Option<$t> {
                (*self).checked_add(*by)
            }
            #[inline]
            fn steps_between(_a: &$t, _b: &$t, _by: &$t) -> Option<usize> {
                None
            }
        }
    )*)
}

step_impl_unsigned!(usize u8 u16 u32);
step_impl_signed!(isize i8 i16 i32);
#[cfg(target_pointer_width = "64")]
step_impl_unsigned!(u64);
#[cfg(target_pointer_width = "64")]
step_impl_signed!(i64);
// If the target pointer width is not 64-bits, we
// assume here that it is less than 64-bits.
#[cfg(not(target_pointer_width = "64"))]
step_impl_no_between!(u64 i64);

/// An adapter for stepping range iterators by a custom amount.
///
/// The resulting iterator handles overflow by stopping. The `A`
/// parameter is the type being iterated over, while `R` is the range
/// type (usually one of `std::ops::{Range, RangeFrom}`.
#[derive(Clone)]
#[unstable(feature = "step_by", reason = "recent addition",
           issue = "27741")]
pub struct StepBy<A, R> {
    step_by: A,
    range: R,
}

impl<A: Step> RangeFrom<A> {
    /// Creates an iterator starting at the same point, but stepping by
    /// the given amount at each iteration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for i in (0u8..).step_by(2) {
    ///     println!("{}", i);
    /// }
    /// ```
    ///
    /// This prints all even `u8` values.
    #[unstable(feature = "step_by", reason = "recent addition",
               issue = "27741")]
    pub fn step_by(self, by: A) -> StepBy<A, Self> {
        StepBy {
            step_by: by,
            range: self
        }
    }
}

impl<A: Step> ops::Range<A> {
    /// Creates an iterator with the same range, but stepping by the
    /// given amount at each iteration.
    ///
    /// The resulting iterator handles overflow by stopping.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(step_by)]
    ///
    /// for i in (0..10).step_by(2) {
    ///     println!("{}", i);
    /// }
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// 0
    /// 2
    /// 4
    /// 6
    /// 8
    /// ```
    #[unstable(feature = "step_by", reason = "recent addition",
               issue = "27741")]
    pub fn step_by(self, by: A) -> StepBy<A, Self> {
        StepBy {
            step_by: by,
            range: self
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Iterator for StepBy<A, RangeFrom<A>> where
    A: Clone,
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut n = &self.range.start + &self.step_by;
        mem::swap(&mut n, &mut self.range.start);
        Some(n)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None) // Too bad we can't specify an infinite lower bound
    }
}

/// An iterator over the range [start, stop]
#[derive(Clone)]
#[unstable(feature = "range_inclusive",
           reason = "likely to be replaced by range notation and adapters",
           issue = "27777")]
pub struct RangeInclusive<A> {
    range: ops::Range<A>,
    done: bool,
}

/// Returns an iterator over the range [start, stop].
#[inline]
#[unstable(feature = "range_inclusive",
           reason = "likely to be replaced by range notation and adapters",
           issue = "27777")]
pub fn range_inclusive<A>(start: A, stop: A) -> RangeInclusive<A>
    where A: Step + One + Clone
{
    RangeInclusive {
        range: start..stop,
        done: false,
    }
}

#[unstable(feature = "range_inclusive",
           reason = "likely to be replaced by range notation and adapters",
           issue = "27777")]
impl<A> Iterator for RangeInclusive<A> where
    A: PartialEq + Step + One + Clone,
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.range.next().or_else(|| {
            if !self.done && self.range.start == self.range.end {
                self.done = true;
                Some(self.range.end.clone())
            } else {
                None
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.range.size_hint();
        if self.done {
            (lo, hi)
        } else {
            let lo = lo.saturating_add(1);
            let hi = hi.and_then(|x| x.checked_add(1));
            (lo, hi)
        }
    }
}

#[unstable(feature = "range_inclusive",
           reason = "likely to be replaced by range notation and adapters",
           issue = "27777")]
impl<A> DoubleEndedIterator for RangeInclusive<A> where
    A: PartialEq + Step + One + Clone,
    for<'a> &'a A: Add<&'a A, Output = A>,
    for<'a> &'a A: Sub<Output=A>
{
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.range.end > self.range.start {
            let result = self.range.end.clone();
            self.range.end = &self.range.end - &A::one();
            Some(result)
        } else if !self.done && self.range.start == self.range.end {
            self.done = true;
            Some(self.range.end.clone())
        } else {
            None
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step + Zero + Clone> Iterator for StepBy<A, ops::Range<A>> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let rev = self.step_by < A::zero();
        if (rev && self.range.start > self.range.end) ||
           (!rev && self.range.start < self.range.end)
        {
            match self.range.start.step(&self.step_by) {
                Some(mut n) => {
                    mem::swap(&mut self.range.start, &mut n);
                    Some(n)
                },
                None => {
                    let mut n = self.range.end.clone();
                    mem::swap(&mut self.range.start, &mut n);
                    Some(n)
                }
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match Step::steps_between(&self.range.start,
                                  &self.range.end,
                                  &self.step_by) {
            Some(hint) => (hint, Some(hint)),
            None       => (0, None)
        }
    }
}

macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl ExactSizeIterator for ops::Range<$t> { }
    )*)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step + One> Iterator for ops::Range<A> where
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.start < self.end {
            let mut n = &self.start + &A::one();
            mem::swap(&mut n, &mut self.start);
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match Step::steps_between(&self.start, &self.end, &A::one()) {
            Some(hint) => (hint, Some(hint)),
            None => (0, None)
        }
    }
}

// Ranges of u64 and i64 are excluded because they cannot guarantee having
// a length <= usize::MAX, which is required by ExactSizeIterator.
range_exact_iter_impl!(usize u8 u16 u32 isize i8 i16 i32);

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step + One + Clone> DoubleEndedIterator for ops::Range<A> where
    for<'a> &'a A: Add<&'a A, Output = A>,
    for<'a> &'a A: Sub<&'a A, Output = A>
{
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.start < self.end {
            self.end = &self.end - &A::one();
            Some(self.end.clone())
        } else {
            None
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step + One> Iterator for ops::RangeFrom<A> where
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut n = &self.start + &A::one();
        mem::swap(&mut n, &mut self.start);
        Some(n)
    }
}

/// An iterator that repeats an element endlessly
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Repeat<A> {
    element: A
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> Iterator for Repeat<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> { Some(self.element.clone()) }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { (usize::MAX, None) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> DoubleEndedIterator for Repeat<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { Some(self.element.clone()) }
}

/// Creates a new iterator that endlessly repeats a single element.
///
/// The `repeat()` function repeats a single value over and over and over and
/// over and over and 🔁.
///
/// Infinite iterators like `repeat()` are often used with adapters like
/// [`take()`], in order to make them finite.
///
/// [`take()`]: trait.Iterator.html#method.take
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // the number four 4ever:
/// let mut fours = iter::repeat(4);
///
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
/// assert_eq!(Some(4), fours.next());
///
/// // yup, still four
/// assert_eq!(Some(4), fours.next());
/// ```
///
/// Going finite with [`take()`]:
///
/// ```
/// use std::iter;
///
/// // that last example was too many fours. Let's only have four fours.
/// let mut four_fours = iter::repeat(4).take(4);
///
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
/// assert_eq!(Some(4), four_fours.next());
///
/// // ... and now we're done
/// assert_eq!(None, four_fours.next());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn repeat<T: Clone>(elt: T) -> Repeat<T> {
    Repeat{element: elt}
}

/// An iterator that yields nothing.
#[stable(feature = "iter_empty", since = "1.2.0")]
pub struct Empty<T>(marker::PhantomData<T>);

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Iterator for Empty<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        (0, Some(0))
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> DoubleEndedIterator for Empty<T> {
    fn next_back(&mut self) -> Option<T> {
        None
    }
}

#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> ExactSizeIterator for Empty<T> {
    fn len(&self) -> usize {
        0
    }
}

// not #[derive] because that adds a Clone bound on T,
// which isn't necessary.
#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Clone for Empty<T> {
    fn clone(&self) -> Empty<T> {
        Empty(marker::PhantomData)
    }
}

// not #[derive] because that adds a Default bound on T,
// which isn't necessary.
#[stable(feature = "iter_empty", since = "1.2.0")]
impl<T> Default for Empty<T> {
    fn default() -> Empty<T> {
        Empty(marker::PhantomData)
    }
}

/// Creates an iterator that yields nothing.
///
/// # Exampes
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // this could have been an iterator over i32, but alas, it's just not.
/// let mut nope = iter::empty::<i32>();
///
/// assert_eq!(None, nope.next());
/// ```
#[stable(feature = "iter_empty", since = "1.2.0")]
pub fn empty<T>() -> Empty<T> {
    Empty(marker::PhantomData)
}

/// An iterator that yields an element exactly once.
#[derive(Clone)]
#[stable(feature = "iter_once", since = "1.2.0")]
pub struct Once<T> {
    inner: ::option::IntoIter<T>
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> Iterator for Once<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> DoubleEndedIterator for Once<T> {
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
    }
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> ExactSizeIterator for Once<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Creates an iterator that yields an element exactly once.
///
/// This is commonly used to adapt a single value into a [`chain()`] of other
/// kinds of iteration. Maybe you have an iterator that covers almost
/// everything, but you need an extra special case. Maybe you have a function
/// which works on iterators, but you only need to process one value.
///
/// [`chain()`]: trait.Iterator.html#method.chain
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // one is the loneliest number
/// let mut one = iter::once(1);
///
/// assert_eq!(Some(1), one.next());
///
/// // just one, that's all we get
/// assert_eq!(None, one.next());
/// ```
///
/// Chaining together with another iterator. Let's say that we want to iterate
/// over each file of the `.foo` directory, but also a configuration file,
/// `.foorc`:
///
/// ```no_run
/// use std::iter;
/// use std::fs;
/// use std::path::PathBuf;
///
/// let dirs = fs::read_dir(".foo").unwrap();
///
/// // we need to convert from an iterator of DirEntry-s to an iterator of
/// // PathBufs, so we use map
/// let dirs = dirs.map(|file| file.unwrap().path());
///
/// // now, our iterator just for our config file
/// let config = iter::once(PathBuf::from(".foorc"));
///
/// // chain the two iterators together into one big iterator
/// let files = dirs.chain(config);
///
/// // this will give us all of the files in .foo as well as .foorc
/// for f in files {
///     println!("{:?}", f);
/// }
/// ```
#[stable(feature = "iter_once", since = "1.2.0")]
pub fn once<T>(value: T) -> Once<T> {
    Once { inner: Some(value).into_iter() }
}

/// Functions for lexicographical ordering of sequences.
///
/// Lexicographical ordering through `<`, `<=`, `>=`, `>` requires
/// that the elements implement both `PartialEq` and `PartialOrd`.
///
/// If two sequences are equal up until the point where one ends,
/// the shorter sequence compares less.
#[deprecated(since = "1.4.0", reason = "use the equivalent methods on `Iterator` instead")]
#[unstable(feature = "iter_order", reason = "needs review and revision",
           issue = "27737")]
pub mod order {
    use cmp;
    use cmp::{Eq, Ord, PartialOrd, PartialEq};
    use option::Option;
    use super::Iterator;

    /// Compare `a` and `b` for equality using `Eq`
    pub fn equals<A, L, R>(a: L, b: R) -> bool where
        A: Eq,
        L: Iterator<Item=A>,
        R: Iterator<Item=A>,
    {
        a.eq(b)
    }

    /// Order `a` and `b` lexicographically using `Ord`
    pub fn cmp<A, L, R>(a: L, b: R) -> cmp::Ordering where
        A: Ord,
        L: Iterator<Item=A>,
        R: Iterator<Item=A>,
    {
        a.cmp(b)
    }

    /// Order `a` and `b` lexicographically using `PartialOrd`
    pub fn partial_cmp<L: Iterator, R: Iterator>(a: L, b: R) -> Option<cmp::Ordering> where
        L::Item: PartialOrd<R::Item>
    {
        a.partial_cmp(b)
    }

    /// Compare `a` and `b` for equality (Using partial equality, `PartialEq`)
    pub fn eq<L: Iterator, R: Iterator>(a: L, b: R) -> bool where
        L::Item: PartialEq<R::Item>,
    {
        a.eq(b)
    }

    /// Compares `a` and `b` for nonequality (Using partial equality, `PartialEq`)
    pub fn ne<L: Iterator, R: Iterator>(a: L, b: R) -> bool where
        L::Item: PartialEq<R::Item>,
    {
        a.ne(b)
    }

    /// Returns `a` < `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn lt<L: Iterator, R: Iterator>(a: L, b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        a.lt(b)
    }

    /// Returns `a` <= `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn le<L: Iterator, R: Iterator>(a: L, b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        a.le(b)
    }

    /// Returns `a` > `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn gt<L: Iterator, R: Iterator>(a: L, b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        a.gt(b)
    }

    /// Returns `a` >= `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn ge<L: Iterator, R: Iterator>(a: L, b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        a.ge(b)
    }
}
