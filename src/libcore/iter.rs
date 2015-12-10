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
//!     count: usize,
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
//!     // we will be counting with usize
//!     type Item = usize;
//!
//!     // next() is the only required method
//!     fn next(&mut self) -> Option<usize> {
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
//! impl<I: Iterator> IntoIterator for I
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
//! # #![allow(unused_must_use)]
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

/// An interface for dealing with iterators.
///
/// This is the main iterator trait. For more about the concept of iterators
/// generally, please see the [module-level documentation]. In particular, you
/// may want to know how to [implement `Iterator`][impl].
///
/// [module-level documentation]: index.html
/// [impl]: index.html#implementing-iterator
#[lang = "iterator"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "`{Self}` is not an iterator; maybe try calling \
                            `.iter()` or a similar method"]
pub trait Iterator {
    /// The type of the elements being iterated over.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Advances the iterator and returns the next value.
    ///
    /// Returns `None` when iteration is finished. Individual iterator
    /// implementations may choose to resume iteration, and so calling `next()`
    /// again may or may not eventually start returning `Some(Item)` again at some
    /// point.
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
    /// The second half of the tuple that is returned is an `Option<usize>`. A
    /// `None` here means that either there is no known upper bound, or the
    /// upper bound is larger than `usize`.
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
    /// This method will evaluate the iterator until its [`next()`] returns
    /// `None`. Once `None` is encountered, `count()` returns the number of
    /// times it called [`next()`].
    ///
    /// [`next()`]: #method.next
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
    /// This function might panic if the iterator has more than `usize::MAX`
    /// elements.
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
    #[stable(feature = "rust1", since = "1.0.0")]
    fn count(self) -> usize where Self: Sized {
        // Might overflow.
        self.fold(0, |cnt, _| cnt + 1)
    }

    /// Consumes the iterator, returning the last element.
    ///
    /// This method will evaluate the iterator until it returns `None`. While
    /// doing so, it keeps track of the current element. After `None` is
    /// returned, `last()` will then return the last element it saw.
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

    /// Consumes the `n` first elements of the iterator, then returns the
    /// `next()` one.
    ///
    /// This method will evaluate the iterator `n` times, discarding those elements.
    /// After it does so, it will call [`next()`] and return its value.
    ///
    /// [`next()`]: #method.next
    ///
    /// Like most indexing operations, the count starts from zero, so `nth(0)`
    /// returns the first value, `nth(1)` the second, and so on.
    ///
    /// `nth()` will return `None` if `n` is larger than the length of the
    /// iterator.
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
    /// Returning `None` if there are less than `n` elements:
    ///
    /// ```
    /// let a = [1, 2, 3];
    /// assert_eq!(a.iter().nth(10), None);
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
    /// In other words, it zips two iterators together, into a single one. 🤐
    ///
    /// When either iterator returns `None`, all further calls to `next()`
    /// will return `None`.
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
    /// This works because the finite iterator will eventually return `None`,
    /// ending the zipper. Zipping with `(0..)` can look a lot like [`enumerate()`]:
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
    /// [`enumerate()`]: trait.Iterator.html#method.enumerate
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn zip<U>(self, other: U) -> Zip<Self, U::IntoIter> where
        Self: Sized, U: IntoIterator
    {
        Zip{a: self, b: other.into_iter()}
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
    /// [`for`]: ../../book/loops.html#for
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
    /// The closure must return an [`Option<T>`]. `filter_map()` creates an
    /// iterator which calls this closure on each element. If the closure
    /// returns `Some(element)`, then that element is returned. If the
    /// closure returns `None`, it will try again, and call the closure on the
    /// next element, seeing if it will return `Some`.
    ///
    /// [`Option<T>`]: ../option/enum.Option.html
    ///
    /// Why `filter_map()` and not just [`filter()`].[`map()`]? The key is in this
    /// part:
    ///
    /// [`filter()`]: #method.filter
    /// [`map()`]: #method.map
    ///
    /// > If the closure returns `Some(element)`, then that element is returned.
    ///
    /// In other words, it removes the [`Option<T>`] layer automatically. If your
    /// mapping is already returning an [`Option<T>`] and you want to skip over
    /// `None`s, then `filter_map()` is much, much nicer to use.
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
    /// Here's the same example, but with [`filter()`] and [`map()`]:
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
    /// There's an extra layer of `Some` in there.
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
    /// different sized integer, the [`zip()`] function provides similar
    /// functionality.
    ///
    /// [`usize`]: ../primitive.usize.html
    /// [`zip()`]: #method.zip
    ///
    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so enumerating more than
    /// [`usize::MAX`] elements either produces the wrong result or panics. If
    /// debug assertions are enabled, a panic is guaranteed.
    ///
    /// [`usize::MAX`]: ../usize/constant.MAX.html
    ///
    /// # Panics
    ///
    /// The returned iterator might panic if the to-be-returned index would
    /// overflow a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter().enumerate();
    ///
    /// assert_eq!(iter.next(), Some((0, &1)));
    /// assert_eq!(iter.next(), Some((1, &2)));
    /// assert_eq!(iter.next(), Some((2, &3)));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn enumerate(self) -> Enumerate<Self> where Self: Sized {
        Enumerate { iter: self, count: 0 }
    }

    /// Creates an iterator which can look at the `next()` element without
    /// consuming it.
    ///
    /// Adds a [`peek()`] method to an iterator. See its documentation for
    /// more information.
    ///
    /// [`peek()`]: struct.Peekable.html#method.peek
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

    /// Creates an iterator that [`skip()`]s elements based on a predicate.
    ///
    /// [`skip()`]: #method.skip
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

    /// An iterator similar to `fold()`, with internal state.
    ///
    /// `scan()` accumulates a final value, similar to [`fold()`], but instead
    /// of passing along an accumulator, it maintains the accumulator internally.
    ///
    /// [`fold()`]: #method.fold
    ///
    /// On each iteraton of `scan()`, you can assign to the internal state, and
    /// a mutable reference to the state is passed as the first argument to the
    /// closure, allowing you to modify it on each iteration.
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
    /// The [`map()`] adapter is very useful, but only when the closure
    /// argument produces values. If it produces an iterator instead, there's
    /// an extra layer of indirection. `flat_map()` will remove this extra layer
    /// on its own.
    ///
    /// [`map()`]: #method.map
    ///
    /// Another way of thinking about `flat_map()`: [`map()`]'s closure returns
    /// one item for each element, and `flat_map()`'s closure returns an
    /// iterator for each element.
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

    /// Creates an iterator which ends after the first `None`.
    ///
    /// After an iterator returns `None`, future calls may or may not yield
    /// `Some(T)` again. `fuse()` adapts an iterator, ensuring that after a
    /// `None` is given, it will always return `None` forever.
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
    /// collection into another. You take a collection, call `iter()` on it,
    /// do a bunch of transformations, and then `collect()` at the end.
    ///
    /// One of the keys to `collect()`'s power is that many things you might
    /// not think of as 'collections' actually are. For example, a [`String`]
    /// is a collection of [`char`]s. And a collection of [`Result<T, E>`] can
    /// be thought of as single [`Result<Collection<T>, E>`]. See the examples
    /// below for more.
    ///
    /// [`String`]: ../string/struct.String.html
    /// [`Result<T, E>`]: ../result/enum.Result.html
    /// [`char`]: ../primitive.char.html
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
    /// [`VecDeque<T>`]: ../collections/struct.VecDeque.html
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
    /// Using the 'turbofish' instead of annotationg `doubled`:
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
    /// If you have a list of [`Result<T, E>`]s, you can use `collect()` to
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
    /// arguments: an 'accumulator', and an element. It returns the value that
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
    /// `all()` is short-circuting; in other words, it will stop processing
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
    /// `any()` is short-circuting; in other words, it will stop processing
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
    /// `true`, then `find()` returns `Some(element)`. If they all return
    /// `false`, it returns `None`.
    ///
    /// `find()` is short-circuting; in other words, it will stop processing
    /// as soon as the closure returns `true`.
    ///
    /// Because `find()` takes a reference, and many iterators iterate over
    /// references, this leads to a possibly confusing situation where the
    /// argument is a double reference. You can see this effect in the
    /// examples below, with `&&x`.
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
    /// returns `true`, then `position()` returns `Some(index)`. If all of
    /// them return `false`, it returns `None`.
    ///
    /// `position()` is short-circuting; in other words, it will stop
    /// processing as soon as it finds a `true`.
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
    /// This function might panic if the iterator has more than `usize::MAX`
    /// non-matching elements.
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
    /// let a = [1, 2, 3];
    ///
    /// let mut iter = a.iter();
    ///
    /// assert_eq!(iter.position(|&x| x == 2), Some(1));
    ///
    /// // we can still use `iter`, as there are more elements.
    /// assert_eq!(iter.next(), Some(&3));
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
    /// `Some(index)`. If all of them return `false`, it returns `None`.
    ///
    /// `rposition()` is short-circuting; in other words, it will stop
    /// processing as soon as it finds a `true`.
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
            if predicate(v) {
                return Some(i - 1);
            }
            // No need for an overflow check here, because `ExactSizeIterator`
            // implies that the number of elements fits into a `usize`.
            i -= 1;
        }
        None
    }

    /// Returns the maximum element of an iterator.
    ///
    /// If the two elements are equally maximum, the latest element is
    /// returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
    /// assert_eq!(a.iter().max(), Some(&3));
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
    /// If the two elements are equally minimum, the first element is
    /// returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let a = [1, 2, 3];
    ///
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

    #[allow(missing_docs)]
    #[inline]
    #[unstable(feature = "iter_cmp",
               reason = "may want to produce an Ordering directly; see #15311",
               issue = "27724")]
    #[rustc_deprecated(reason = "renamed to max_by_key", since = "1.6.0")]
    fn max_by<B: Ord, F>(self, f: F) -> Option<Self::Item> where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        self.max_by_key(f)
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

    #[inline]
    #[allow(missing_docs)]
    #[unstable(feature = "iter_cmp",
               reason = "may want to produce an Ordering directly; see #15311",
               issue = "27724")]
    #[rustc_deprecated(reason = "renamed to min_by_key", since = "1.6.0")]
    fn min_by<B: Ord, F>(self, f: F) -> Option<Self::Item> where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
    {
        self.min_by_key(f)
    }

    /// Returns the element that gives the minimum value from the
    /// specified function.
    ///
    /// Returns the latest element if the comparison determines two elements
    /// to be equally minimum.
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
    /// This function is, in some sense, the opposite of [`zip()`].
    ///
    /// [`zip()`]: #method.zip
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

    /// Creates an iterator which clone()s all of its elements.
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
    /// Instead of stopping at `None`, the iterator will instead start again,
    /// from the beginning. After iterating again, it will start at the
    /// beginning again. And again. And again. Forever.
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
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(iter_arith)]
    ///
    /// let a = [1, 2, 3];
    /// let sum: i32 = a.iter().sum();
    ///
    /// assert_eq!(sum, 6);
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
    /// An empty iterator returns the one value of the type.
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

/// Conversion from an `Iterator`.
///
/// By implementing `FromIterator` for a type, you define how it will be
/// created from an iterator. This is common for types which describe a
/// collection of some kind.
///
/// `FromIterator`'s [`from_iter()`] is rarely called explicitly, and is instead
/// used through [`Iterator`]'s [`collect()`] method. See [`collect()`]'s
/// documentation for more examples.
///
/// [`from_iter()`]: #tymethod.from_iter
/// [`Iterator`]: trait.Iterator.html
/// [`collect()`]: trait.Iterator.html#method.collect
///
/// See also: [`IntoIterator`].
///
/// [`IntoIterator`]: trait.IntoIterator.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter::FromIterator;
///
/// let five_fives = std::iter::repeat(5).take(5);
///
/// let v = Vec::from_iter(five_fives);
///
/// assert_eq!(v, vec![5, 5, 5, 5, 5]);
/// ```
///
/// Using [`collect()`] to implicitly use `FromIterator`:
///
/// ```
/// let five_fives = std::iter::repeat(5).take(5);
///
/// let v: Vec<i32> = five_fives.collect();
///
/// assert_eq!(v, vec![5, 5, 5, 5, 5]);
/// ```
///
/// Implementing `FromIterator` for your type:
///
/// ```
/// use std::iter::FromIterator;
///
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // and we'll implement FromIterator
/// impl FromIterator<i32> for MyCollection {
///     fn from_iter<I: IntoIterator<Item=i32>>(iterator: I) -> Self {
///         let mut c = MyCollection::new();
///
///         for i in iterator {
///             c.add(i);
///         }
///
///         c
///     }
/// }
///
/// // Now we can make a new iterator...
/// let iter = (0..5).into_iter();
///
/// // ... and make a MyCollection out of it
/// let c = MyCollection::from_iter(iter);
///
/// assert_eq!(c.0, vec![0, 1, 2, 3, 4]);
///
/// // collect works too!
///
/// let iter = (0..5).into_iter();
/// let c: MyCollection = iter.collect();
///
/// assert_eq!(c.0, vec![0, 1, 2, 3, 4]);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented="a collection of type `{Self}` cannot be \
                          built from an iterator over elements of type `{A}`"]
pub trait FromIterator<A>: Sized {
    /// Creates a value from an iterator.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: trait.FromIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::iter::FromIterator;
    ///
    /// let five_fives = std::iter::repeat(5).take(5);
    ///
    /// let v = Vec::from_iter(five_fives);
    ///
    /// assert_eq!(v, vec![5, 5, 5, 5, 5]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_iter<T: IntoIterator<Item=A>>(iterator: T) -> Self;
}

/// Conversion into an `Iterator`.
///
/// By implementing `IntoIterator` for a type, you define how it will be
/// converted to an iterator. This is common for types which describe a
/// collection of some kind.
///
/// One benefit of implementing `IntoIterator` is that your type will [work
/// with Rust's `for` loop syntax](index.html#for-loops-and-intoiterator).
///
/// See also: [`FromIterator`].
///
/// [`FromIterator`]: trait.FromIterator.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let v = vec![1, 2, 3];
///
/// let mut iter = v.into_iter();
///
/// let n = iter.next();
/// assert_eq!(Some(1), n);
///
/// let n = iter.next();
/// assert_eq!(Some(2), n);
///
/// let n = iter.next();
/// assert_eq!(Some(3), n);
///
/// let n = iter.next();
/// assert_eq!(None, n);
/// ```
///
/// Implementing `IntoIterator` for your type:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // and we'll implement IntoIterator
/// impl IntoIterator for MyCollection {
///     type Item = i32;
///     type IntoIter = ::std::vec::IntoIter<i32>;
///
///     fn into_iter(self) -> Self::IntoIter {
///         self.0.into_iter()
///     }
/// }
///
/// // Now we can make a new collection...
/// let mut c = MyCollection::new();
///
/// // ... add some stuff to it ...
/// c.add(0);
/// c.add(1);
/// c.add(2);
///
/// // ... and then turn it into an Iterator:
/// for (i, n) in c.into_iter().enumerate() {
///     assert_eq!(i as i32, n);
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IntoIterator {
    /// The type of the elements being iterated over.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Which kind of iterator are we turning this into?
    #[stable(feature = "rust1", since = "1.0.0")]
    type IntoIter: Iterator<Item=Self::Item>;

    /// Creates an iterator from a value.
    ///
    /// See the [module-level documentation] for more.
    ///
    /// [module-level documentation]: trait.IntoIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v = vec![1, 2, 3];
    ///
    /// let mut iter = v.into_iter();
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(1), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(2), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(3), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(None, n);
    /// ```
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

/// Extend a collection with the contents of an iterator.
///
/// Iterators produce a series of values, and collections can also be thought
/// of as a series of values. The `Extend` trait bridges this gap, allowing you
/// to extend a collection by including the contents of that iterator.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // You can extend a String with some chars:
/// let mut message = String::from("The first three letters are: ");
///
/// message.extend(&['a', 'b', 'c']);
///
/// assert_eq!("abc", &message[29..32]);
/// ```
///
/// Implementing `Extend`:
///
/// ```
/// // A sample collection, that's just a wrapper over Vec<T>
/// #[derive(Debug)]
/// struct MyCollection(Vec<i32>);
///
/// // Let's give it some methods so we can create one and add things
/// // to it.
/// impl MyCollection {
///     fn new() -> MyCollection {
///         MyCollection(Vec::new())
///     }
///
///     fn add(&mut self, elem: i32) {
///         self.0.push(elem);
///     }
/// }
///
/// // since MyCollection has a list of i32s, we implement Extend for i32
/// impl Extend<i32> for MyCollection {
///
///     // This is a bit simpler with the concrete type signature: we can call
///     // extend on anything which can be turned into an Iterator which gives
///     // us i32s. Because we need i32s to put into MyCollection.
///     fn extend<T: IntoIterator<Item=i32>>(&mut self, iterable: T) {
///
///         // The implementation is very straightforward: loop through the
///         // iterator, and add() each element to ourselves.
///         for elem in iterable {
///             self.add(elem);
///         }
///     }
/// }
///
/// let mut c = MyCollection::new();
///
/// c.add(5);
/// c.add(6);
/// c.add(7);
///
/// // let's extend our collection with three more numbers
/// c.extend(vec![1, 2, 3]);
///
/// // we've added these elements onto the end
/// assert_eq!("MyCollection([5, 6, 7, 1, 2, 3])", format!("{:?}", c));
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Extend<A> {
    /// Extends a collection with the contents of an iterator.
    ///
    /// As this is the only method for this trait, the [trait-level] docs
    /// contain more details.
    ///
    /// [trait-level]: trait.Extend.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // You can extend a String with some chars:
    /// let mut message = String::from("abc");
    ///
    /// message.extend(['d', 'e', 'f'].iter());
    ///
    /// assert_eq!("abcdef", &message);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn extend<T: IntoIterator<Item=A>>(&mut self, iterable: T);
}

/// An iterator able to yield elements from both ends.
///
/// Something that implements `DoubleEndedIterator` has one extra capability
/// over something that implements [`Iterator`]: the ability to also take
/// `Item`s from the back, as well as the front.
///
/// It is important to note that both back and forth work on the same range,
/// and do not cross: iteration is over when they meet in the middle.
///
/// [`Iterator`]: trait.Iterator.html
/// # Examples
///
/// Basic usage:
///
/// ```
/// let numbers = vec![1, 2, 3];
///
/// let mut iter = numbers.iter();
///
/// let n = iter.next();
/// assert_eq!(Some(&1), n);
///
/// let n = iter.next_back();
/// assert_eq!(Some(&3), n);
///
/// let n = iter.next_back();
/// assert_eq!(Some(&2), n);
///
/// let n = iter.next();
/// assert_eq!(None, n);
///
/// let n = iter.next_back();
/// assert_eq!(None, n);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait DoubleEndedIterator: Iterator {
    /// An iterator able to yield elements from both ends.
    ///
    /// As this is the only method for this trait, the [trait-level] docs
    /// contain more details.
    ///
    /// [trait-level]: trait.DoubleEndedIterator.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let numbers = vec![1, 2, 3];
    ///
    /// let mut iter = numbers.iter();
    ///
    /// let n = iter.next();
    /// assert_eq!(Some(&1), n);
    ///
    /// let n = iter.next_back();
    /// assert_eq!(Some(&3), n);
    ///
    /// let n = iter.next_back();
    /// assert_eq!(Some(&2), n);
    ///
    /// let n = iter.next();
    /// assert_eq!(None, n);
    ///
    /// let n = iter.next_back();
    /// assert_eq!(None, n);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next_back(&mut self) -> Option<Self::Item>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for &'a mut I {
    fn next_back(&mut self) -> Option<I::Item> { (**self).next_back() }
}

/// An iterator that knows its exact length.
///
/// Many [`Iterator`]s don't know how many times they will iterate, but some do.
/// If an iterator knows how many times it can iterate, providing access to
/// that information can be useful. For example, if you want to iterate
/// backwards, a good start is to know where the end is.
///
/// When implementing an `ExactSizeIterator`, You must also implement
/// [`Iterator`]. When doing so, the implementation of [`size_hint()`] *must*
/// return the exact size of the iterator.
///
/// [`Iterator`]: trait.Iterator.html
/// [`size_hint()`]: trait.Iterator.html#method.size_hint
///
/// The [`len()`] method has a default implementation, so you usually shouldn't
/// implement it. However, you may be able to provide a more performant
/// implementation than the default, so overriding it in this case makes sense.
///
/// [`len()`]: #method.len
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // a finite range knows exactly how many times it will iterate
/// let five = 0..5;
///
/// assert_eq!(5, five.len());
/// ```
///
/// In the [module level docs][moddocs], we implemented an [`Iterator`],
/// `Counter`. Let's implement `ExactSizeIterator` for it as well:
///
/// [moddocs]: index.html
///
/// ```
/// # struct Counter {
/// #     count: usize,
/// # }
/// # impl Counter {
/// #     fn new() -> Counter {
/// #         Counter { count: 0 }
/// #     }
/// # }
/// # impl Iterator for Counter {
/// #     type Item = usize;
/// #     fn next(&mut self) -> Option<usize> {
/// #         self.count += 1;
/// #         if self.count < 6 {
/// #             Some(self.count)
/// #         } else {
/// #             None
/// #         }
/// #     }
/// # }
/// impl ExactSizeIterator for Counter {
///     // We already have the number of iterations, so we can use it directly.
///     fn len(&self) -> usize {
///         self.count
///     }
/// }
///
/// // And now we can use it!
///
/// let counter = Counter::new();
///
/// assert_eq!(0, counter.len());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ExactSizeIterator: Iterator {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Returns the exact number of times the iterator will iterate.
    ///
    /// This method has a default implementation, so you usually should not
    /// implement it directly. However, if you can provide a more efficient
    /// implementation, you can do so. See the [trait-level] docs for an
    /// example.
    ///
    /// This function has the same safety guarantees as the [`size_hint()`]
    /// function.
    ///
    /// [trait-level]: trait.ExactSizeIterator.html
    /// [`size_hint()`]: trait.Iterator.html#method.size_hint
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // a finite range knows exactly how many times it will iterate
    /// let five = 0..5;
    ///
    /// assert_eq!(5, five.len());
    /// ```
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

/// An double-ended iterator with the direction inverted.
///
/// This `struct` is created by the [`rev()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`rev()`]: trait.Iterator.html#method.rev
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that clones the elements of an underlying iterator.
///
/// This `struct` is created by the [`cloned()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`cloned()`]: trait.Iterator.html#method.cloned
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that repeats endlessly.
///
/// This `struct` is created by the [`cycle()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`cycle()`]: trait.Iterator.html#method.cycle
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that strings two iterators together.
///
/// This `struct` is created by the [`chain()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`chain()`]: trait.Iterator.html#method.chain
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that iterates two other iterators simultaneously.
///
/// This `struct` is created by the [`zip()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`zip()`]: trait.Iterator.html#method.zip
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that maps the values of `iter` with `f`.
///
/// This `struct` is created by the [`map()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`map()`]: trait.Iterator.html#method.map
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that filters the elements of `iter` with `predicate`.
///
/// This `struct` is created by the [`filter()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`filter()`]: trait.Iterator.html#method.filter
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that uses `f` to both filter and map elements from `iter`.
///
/// This `struct` is created by the [`filter_map()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`filter_map()`]: trait.Iterator.html#method.filter_map
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that yields the current count and the element during iteration.
///
/// This `struct` is created by the [`enumerate()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`enumerate()`]: trait.Iterator.html#method.enumerate
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator with a `peek()` that returns an optional reference to the next
/// element.
///
/// This `struct` is created by the [`peekable()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`peekable()`]: trait.Iterator.html#method.peekable
/// [`Iterator`]: trait.Iterator.html
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

impl<I: Iterator> Peekable<I> {
    /// Returns a reference to the next() value without advancing the iterator.
    ///
    /// The `peek()` method will return the value that a call to [`next()`] would
    /// return, but does not advance the iterator. Like [`next()`], if there is
    /// a value, it's wrapped in a `Some(T)`, but if the iterator is over, it
    /// will return `None`.
    ///
    /// [`next()`]: trait.Iterator.html#tymethod.next
    ///
    /// Because `peek()` returns reference, and many iterators iterate over
    /// references, this leads to a possibly confusing situation where the
    /// return value is a double reference. You can see this effect in the
    /// examples below, with `&&i32`.
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
    pub fn peek(&mut self) -> Option<&I::Item> {
        if self.peeked.is_none() {
            self.peeked = self.iter.next();
        }
        match self.peeked {
            Some(ref value) => Some(value),
            None => None,
        }
    }

    /// Checks if the iterator has finished iterating.
    ///
    /// Returns `true` if there are no more elements in the iterator, and
    /// `false` if there are.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(peekable_is_empty)]
    ///
    /// let xs = [1, 2, 3];
    ///
    /// let mut iter = xs.iter().peekable();
    ///
    /// // there are still elements to iterate over
    /// assert_eq!(iter.is_empty(), false);
    ///
    /// // let's consume the iterator
    /// iter.next();
    /// iter.next();
    /// iter.next();
    ///
    /// assert_eq!(iter.is_empty(), true);
    /// ```
    #[unstable(feature = "peekable_is_empty", issue = "27701")]
    #[inline]
    pub fn is_empty(&mut self) -> bool {
        self.peek().is_none()
    }
}

/// An iterator that rejects elements while `predicate` is true.
///
/// This `struct` is created by the [`skip_while()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`skip_while()`]: trait.Iterator.html#method.skip_while
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that only accepts elements while `predicate` is true.
///
/// This `struct` is created by the [`take_while()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`take_while()`]: trait.Iterator.html#method.take_while
/// [`Iterator`]: trait.Iterator.html
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
///
/// This `struct` is created by the [`skip()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`skip()`]: trait.Iterator.html#method.skip
/// [`Iterator`]: trait.Iterator.html
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
///
/// This `struct` is created by the [`take()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`take()`]: trait.Iterator.html#method.take
/// [`Iterator`]: trait.Iterator.html
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


/// An iterator to maintain state while iterating another iterator.
///
/// This `struct` is created by the [`scan()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`scan()`]: trait.Iterator.html#method.scan
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that maps each element to an iterator, and yields the elements
/// of the produced iterators.
///
/// This `struct` is created by the [`flat_map()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`flat_map()`]: trait.Iterator.html#method.flat_map
/// [`Iterator`]: trait.Iterator.html
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
/// This `struct` is created by the [`fuse()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`fuse()`]: trait.Iterator.html#method.fuse
/// [`Iterator`]: trait.Iterator.html
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

/// An iterator that calls a function with a reference to each element before
/// yielding it.
///
/// This `struct` is created by the [`inspect()`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`inspect()`]: trait.Iterator.html#method.inspect
/// [`Iterator`]: trait.Iterator.html
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
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "27741")]
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
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "27741")]
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
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "27741")]
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
#[rustc_deprecated(since = "1.5.0", reason = "replaced with ... syntax")]
#[allow(deprecated)]
pub struct RangeInclusive<A> {
    range: ops::Range<A>,
    done: bool,
}

/// Returns an iterator over the range [start, stop].
#[inline]
#[unstable(feature = "range_inclusive",
           reason = "likely to be replaced by range notation and adapters",
           issue = "27777")]
#[rustc_deprecated(since = "1.5.0", reason = "replaced with ... syntax")]
#[allow(deprecated)]
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
#[rustc_deprecated(since = "1.5.0", reason = "replaced with ... syntax")]
#[allow(deprecated)]
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
#[rustc_deprecated(since = "1.5.0", reason = "replaced with ... syntax")]
#[allow(deprecated)]
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

/// An iterator that repeats an element endlessly.
///
/// This `struct` is created by the [`repeat()`] function. See its documentation for more.
///
/// [`repeat()`]: fn.repeat.html
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
///
/// This `struct` is created by the [`empty()`] function. See its documentation for more.
///
/// [`empty()`]: fn.empty.html
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
/// # Examples
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
///
/// This `struct` is created by the [`once()`] function. See its documentation for more.
///
/// [`once()`]: fn.once.html
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
