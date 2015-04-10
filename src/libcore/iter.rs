// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Composable external iterators
//!
//! # The `Iterator` trait
//!
//! This module defines Rust's core iteration trait. The `Iterator` trait has
//! one unimplemented method, `next`. All other methods are derived through
//! default methods to perform operations such as `zip`, `chain`, `enumerate`,
//! and `fold`.
//!
//! The goal of this module is to unify iteration across all containers in Rust.
//! An iterator can be considered as a state machine which is used to track
//! which element will be yielded next.
//!
//! There are various extensions also defined in this module to assist with
//! various types of iteration, such as the `DoubleEndedIterator` for iterating
//! in reverse, the `FromIterator` trait for creating a container from an
//! iterator, and much more.
//!
//! # Rust's `for` loop
//!
//! The special syntax used by rust's `for` loop is based around the
//! `IntoIterator` trait defined in this module. `for` loops can be viewed as a
//! syntactical expansion into a `loop`, for example, the `for` loop in this
//! example is essentially translated to the `loop` below.
//!
//! ```
//! let values = vec![1, 2, 3];
//!
//! for x in values {
//!     println!("{}", x);
//! }
//!
//! // Rough translation of the iteration without a `for` iterator.
//! # let values = vec![1, 2, 3];
//! let mut it = values.into_iter();
//! loop {
//!     match it.next() {
//!         Some(x) => println!("{}", x),
//!         None => break,
//!     }
//! }
//! ```
//!
//! Because `Iterator`s implement `IntoIterator`, this `for` loop syntax can be applied to any
//! iterator over any type.

#![stable(feature = "rust1", since = "1.0.0")]

use self::MinMaxResult::*;

use clone::Clone;
use cmp;
use cmp::{Ord, PartialOrd, PartialEq};
use default::Default;
use marker;
use mem;
use num::{Int, Zero, One};
use ops::{self, Add, Sub, FnMut, Mul, RangeFrom};
use option::Option::{self, Some, None};
use marker::Sized;
use usize;

fn _assert_is_object_safe(_: &Iterator) {}

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
#[lang="iterator"]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_on_unimplemented = "`{Self}` is not an iterator; maybe try calling \
                            `.iter()` or a similar method"]
pub trait Iterator {
    /// The type of the elements being iterated
    #[stable(feature = "rust1", since = "1.0.0")]
    type Item;

    /// Advance the iterator and return the next value. Return `None` when the
    /// end is reached.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next(&mut self) -> Option<Self::Item>;

    /// Returns a lower and upper bound on the remaining length of the iterator.
    ///
    /// An upper bound of `None` means either there is no known upper bound, or
    /// the upper bound does not fit within a `usize`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn size_hint(&self) -> (usize, Option<usize>) { (0, None) }

    /// Counts the number of elements in this iterator.
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
        self.fold(0, |cnt, _x| cnt + 1)
    }

    /// Loops through the entire iterator, returning the last element.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().last().unwrap() == &5);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn last(self) -> Option<Self::Item> where Self: Sized {
        let mut last = None;
        for x in self { last = Some(x); }
        last
    }

    /// Loops through `n` iterations, returning the `n`th element of the
    /// iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.nth(2).unwrap() == &3);
    /// assert!(it.nth(2) == None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> where Self: Sized {
        for x in self.by_ref() {
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
    /// let mut it = a.iter().chain(b.iter());
    /// assert_eq!(it.next().unwrap(), &0);
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chain<U>(self, other: U) -> Chain<Self, U::IntoIter> where
        Self: Sized, U: IntoIterator<Item=Self::Item>,
    {
        Chain{a: self, b: other.into_iter(), flag: false}
    }

    /// Creates an iterator that iterates over both this and the specified
    /// iterators simultaneously, yielding the two elements as pairs. When
    /// either iterator returns None, all further invocations of next() will
    /// return None.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [0];
    /// let b = [1];
    /// let mut it = a.iter().zip(b.iter());
    /// assert_eq!(it.next().unwrap(), (&0, &1));
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
    /// assert_eq!(it.next().unwrap(), 2);
    /// assert_eq!(it.next().unwrap(), 4);
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
    /// assert_eq!(it.next().unwrap(), &2);
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
    /// If the specified function returns None, the element is skipped.
    /// Otherwise the option is unwrapped and the new value is yielded.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2];
    /// let mut it = a.iter().filter_map(|&x| if x > 1 {Some(2 * x)} else {None});
    /// assert_eq!(it.next().unwrap(), 4);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F> where
        Self: Sized, F: FnMut(Self::Item) -> Option<B>,
    {
        FilterMap { iter: self, f: f }
    }

    /// Creates an iterator that yields a pair of the value returned by this
    /// iterator plus the current index of iteration.
    ///
    /// `enumerate` keeps its count as a `usize`. If you want to count by a
    /// different sized integer, the `zip` function provides similar
    /// functionality.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [100, 200];
    /// let mut it = a.iter().enumerate();
    /// assert_eq!(it.next().unwrap(), (0, &100));
    /// assert_eq!(it.next().unwrap(), (1, &200));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn enumerate(self) -> Enumerate<Self> where Self: Sized {
        Enumerate{iter: self, count: 0}
    }

    /// Creates an iterator that has a `.peek()` method
    /// that returns an optional reference to the next element.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
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
    /// assert_eq!(it.next().unwrap(), &3);
    /// assert_eq!(it.next().unwrap(), &4);
    /// assert_eq!(it.next().unwrap(), &5);
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
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert_eq!(it.next().unwrap(), &2);
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
    /// assert_eq!(it.next().unwrap(), &4);
    /// assert_eq!(it.next().unwrap(), &5);
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
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert_eq!(it.next().unwrap(), &2);
    /// assert_eq!(it.next().unwrap(), &3);
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
    /// from the Scan instance when not None.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().scan(1, |fac, &x| {
    ///   *fac = *fac * x;
    ///   Some(*fac)
    /// });
    /// assert_eq!(it.next().unwrap(), 1);
    /// assert_eq!(it.next().unwrap(), 2);
    /// assert_eq!(it.next().unwrap(), 6);
    /// assert_eq!(it.next().unwrap(), 24);
    /// assert_eq!(it.next().unwrap(), 120);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
        where Self: Sized, F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        Scan{iter: self, f: f, state: initial_state}
    }

    /// Creates an iterator that maps each element to an iterator,
    /// and yields the elements of the produced iterators.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    /// let xs = [2, 3];
    /// let ys = [0, 1, 0, 1, 2];
    /// let it = xs.iter().flat_map(|&x| (0..).take(x));
    /// // Check that `it` has the same elements as `ys`
    /// for (i, x) in it.enumerate() {
    ///     assert_eq!(x, ys[i]);
    /// }
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
    /// # #![feature(core)]
    ///
    /// let a = [1, 4, 2, 3, 8, 9, 6];
    /// let sum: i32 = a.iter()
    ///                 .map(|x| *x)
    ///                 .inspect(|&x| println!("filtering {}", x))
    ///                 .filter(|&x| x % 2 == 0)
    ///                 .inspect(|&x| println!("{} made it through", x))
    ///                 .sum();
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
    /// assert!(partial_sum == 10);
    /// assert!(it.next() == Some(5));
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
    /// # #![feature(core)]
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
                left.extend(Some(x).into_iter())
            } else {
                right.extend(Some(x).into_iter())
            }
        }

        (left, right)
    }

    /// Performs a fold operation over the entire iterator, returning the
    /// eventual state at the end of the iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().fold(0, |acc, &item| acc + item) == 15);
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
        for x in self.by_ref() {
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
    /// # #![feature(core)]
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.any(|x| *x == 3));
    /// assert_eq!(&it[..], [4, 5]);
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn any<F>(&mut self, mut f: F) -> bool where
        Self: Sized,
        F: FnMut(Self::Item) -> bool
    {
        for x in self.by_ref() {
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
    /// # #![feature(core)]
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.find(|&x| *x == 3).unwrap(), &3);
    /// assert_eq!(&it[..], [4, 5]);
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item> where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        for x in self.by_ref() {
            if predicate(&x) { return Some(x) }
        }
        None
    }

    /// Return the index of the first element satisfying the specified predicate
    ///
    /// Does not consume the iterator past the first found element.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.position(|x| *x == 3).unwrap(), 2);
    /// assert_eq!(&it[..], [4, 5]);
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
        Self: Sized,
        P: FnMut(Self::Item) -> bool,
    {
        let mut i = 0;
        for x in self.by_ref() {
            if predicate(x) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Return the index of the last element satisfying the specified predicate
    ///
    /// If no element matches, None is returned.
    ///
    /// Does not consume the iterator *before* the first found element.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    /// let a = [1, 2, 2, 4, 5];
    /// let mut it = a.iter();
    /// assert_eq!(it.rposition(|x| *x == 2).unwrap(), 2);
    /// assert_eq!(&it[..], [1, 2]);
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
    /// assert!(a.iter().max().unwrap() == &5);
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
    /// assert!(a.iter().min().unwrap() == &1);
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

    /// `min_max` finds the minimum and maximum elements in the iterator.
    ///
    /// The return type `MinMaxResult` is an enum of three variants:
    ///
    /// - `NoElements` if the iterator is empty.
    /// - `OneElement(x)` if the iterator has exactly one element.
    /// - `MinMax(x, y)` is returned otherwise, where `x <= y`. Two
    ///    values are equal if and only if there is more than one
    ///    element in the iterator and all elements are equal.
    ///
    /// On an iterator of length `n`, `min_max` does `1.5 * n` comparisons,
    /// and so is faster than calling `min` and `max` separately which does `2 *
    /// n` comparisons.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::iter::MinMaxResult::{NoElements, OneElement, MinMax};
    ///
    /// let a: [i32; 0] = [];
    /// assert_eq!(a.iter().min_max(), NoElements);
    ///
    /// let a = [1];
    /// assert!(a.iter().min_max() == OneElement(&1));
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().min_max() == MinMax(&1, &5));
    ///
    /// let a = [1, 1, 1, 1];
    /// assert!(a.iter().min_max() == MinMax(&1, &1));
    /// ```
    #[unstable(feature = "core", reason = "return type may change")]
    fn min_max(mut self) -> MinMaxResult<Self::Item> where Self: Sized, Self::Item: Ord
    {
        let (mut min, mut max) = match self.next() {
            None => return NoElements,
            Some(x) => {
                match self.next() {
                    None => return OneElement(x),
                    Some(y) => if x <= y {(x, y)} else {(y, x)}
                }
            }
        };

        loop {
            // `first` and `second` are the two next elements we want to look
            // at.  We first compare `first` and `second` (#1). The smaller one
            // is then compared to current minimum (#2). The larger one is
            // compared to current maximum (#3). This way we do 3 comparisons
            // for 2 elements.
            let first = match self.next() {
                None => break,
                Some(x) => x
            };
            let second = match self.next() {
                None => {
                    if first < min {
                        min = first;
                    } else if first >= max {
                        max = first;
                    }
                    break;
                }
                Some(x) => x
            };
            if first <= second {
                if first < min { min = first }
                if second >= max { max = second }
            } else {
                if second < min { min = second }
                if first >= max { max = first }
            }
        }

        MinMax(min, max)
    }

    /// Return the element that gives the maximum value from the
    /// specified function.
    ///
    /// Returns the rightmost element if the comparison determines two elements
    /// to be equally maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    ///
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().max_by(|x| x.abs()).unwrap(), -10);
    /// ```
    #[inline]
    #[unstable(feature = "core",
               reason = "may want to produce an Ordering directly; see #15311")]
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

    /// Return the element that gives the minimum value from the
    /// specified function.
    ///
    /// Returns the leftmost element if the comparison determines two elements
    /// to be equally minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    ///
    /// let a = [-3_i32, 0, 1, 5, -10];
    /// assert_eq!(*a.iter().min_by(|x| x.abs()).unwrap(), 0);
    /// ```
    #[inline]
    #[unstable(feature = "core",
               reason = "may want to produce an Ordering directly; see #15311")]
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
    /// # #![feature(core)]
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
            ts.extend(Some(t).into_iter());
            us.extend(Some(u).into_iter());
        }

        (ts, us)
    }

    /// Creates an iterator that clones the elements it yields. Useful for
    /// converting an Iterator<&T> to an Iterator<T>.
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
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert_eq!(it.next().unwrap(), &2);
    /// assert_eq!(it.next().unwrap(), &1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn cycle(self) -> Cycle<Self> where Self: Sized + Clone {
        Cycle{orig: self.clone(), iter: self}
    }

    /// Use an iterator to reverse a container in place.
    #[unstable(feature = "core",
               reason = "uncertain about placement or widespread use")]
    fn reverse_in_place<'a, T: 'a>(&mut self) where
        Self: Sized + Iterator<Item=&'a mut T> + DoubleEndedIterator
    {
        loop {
            match (self.next(), self.next_back()) {
                (Some(x), Some(y)) => mem::swap(x, y),
                _ => break
            }
        }
    }

    /// Iterates over the entire iterator, summing up all the elements
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    ///
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().cloned();
    /// assert!(it.sum::<i32>() == 15);
    /// ```
    #[unstable(feature="core")]
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
    /// # #![feature(core)]
    ///
    /// fn factorial(n: u32) -> u32 {
    ///     (1..).take_while(|&i| i <= n).product()
    /// }
    /// assert!(factorial(0) == 1);
    /// assert!(factorial(1) == 1);
    /// assert!(factorial(5) == 120);
    /// ```
    #[unstable(feature="core")]
    fn product<P=<Self as Iterator>::Item>(self) -> P where
        P: Mul<Self::Item, Output=P> + One,
        Self: Sized,
    {
        self.fold(One::one(), |p, e| p * e)
    }
}

/// Select an element from an iterator based on the given projection
/// and "comparison" function.
///
/// This is an idiosyncratic helper to try to factor out the
/// commonalities of {max,min}{,_by}. In particular, this avoids
/// having to implement optimisations several times.
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
    /// Build a container with elements from something iterable.
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
/// the [module level documentation](../index.html) for more details.
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
    /// Extend a container with the elements yielded by an arbitrary iterator
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
    /// Yield an element from the end of the range, returning `None` if the
    /// range is empty.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn next_back(&mut self) -> Option<Self::Item>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for &'a mut I {
    fn next_back(&mut self) -> Option<I::Item> { (**self).next_back() }
}

/// An object implementing random access indexing by `usize`
///
/// A `RandomAccessIterator` should be either infinite or a
/// `DoubleEndedIterator`.  Calling `next()` or `next_back()` on a
/// `RandomAccessIterator` reduces the indexable range accordingly. That is,
/// `it.idx(1)` will become `it.idx(0)` after `it.next()` is called.
#[unstable(feature = "core",
           reason = "not widely used, may be better decomposed into Index \
                     and ExactSizeIterator")]
pub trait RandomAccessIterator: Iterator {
    /// Return the number of indexable elements. At most `std::usize::MAX`
    /// elements are indexable, even if the iterator represents a longer range.
    fn indexable(&self) -> usize;

    /// Return an element at an index, or `None` if the index is out of bounds
    fn idx(&mut self, index: usize) -> Option<Self::Item>;
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
    /// Return the exact length of the iterator.
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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<I> RandomAccessIterator for Rev<I>
    where I: DoubleEndedIterator + RandomAccessIterator
{
    #[inline]
    fn indexable(&self) -> usize { self.iter.indexable() }
    #[inline]
    fn idx(&mut self, index: usize) -> Option<<I as Iterator>::Item> {
        let amt = self.indexable();
        if amt > index {
            self.iter.idx(amt - index - 1)
        } else {
            None
        }
    }
}

/// `MinMaxResult` is an enum returned by `min_max`. See `Iterator::min_max` for
/// more detail.
#[derive(Clone, PartialEq, Debug)]
#[unstable(feature = "core",
           reason = "unclear whether such a fine-grained result is widely useful")]
pub enum MinMaxResult<T> {
    /// Empty iterator
    NoElements,

    /// Iterator with one element, so the minimum and maximum are the same
    OneElement(T),

    /// More than one element in the iterator, the first element is not larger
    /// than the second
    MinMax(T, T)
}

impl<T: Clone> MinMaxResult<T> {
    /// `into_option` creates an `Option` of type `(T,T)`. The returned `Option`
    /// has variant `None` if and only if the `MinMaxResult` has variant
    /// `NoElements`. Otherwise variant `Some(x,y)` is returned where `x <= y`.
    /// If `MinMaxResult` has variant `OneElement(x)`, performing this operation
    /// will make one clone of `x`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(core)]
    /// use std::iter::MinMaxResult::{self, NoElements, OneElement, MinMax};
    ///
    /// let r: MinMaxResult<i32> = NoElements;
    /// assert_eq!(r.into_option(), None);
    ///
    /// let r = OneElement(1);
    /// assert_eq!(r.into_option(), Some((1, 1)));
    ///
    /// let r = MinMax(1, 2);
    /// assert_eq!(r.into_option(), Some((1, 2)));
    /// ```
    #[unstable(feature = "core", reason = "type is unstable")]
    pub fn into_option(self) -> Option<(T,T)> {
        match self {
            NoElements => None,
            OneElement(x) => Some((x.clone(), x)),
            MinMax(x, y) => Some((x, y))
        }
    }
}

/// An iterator that clones the elements of an underlying iterator
#[unstable(feature = "core", reason = "recent addition")]
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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<'a, I, T: 'a> RandomAccessIterator for Cloned<I>
    where I: RandomAccessIterator<Item=&'a T>, T: Clone
{
    #[inline]
    fn indexable(&self) -> usize {
        self.it.indexable()
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<T> {
        self.it.idx(index).cloned()
    }
}

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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<I> RandomAccessIterator for Cycle<I> where
    I: Clone + RandomAccessIterator,
{
    #[inline]
    fn indexable(&self) -> usize {
        if self.orig.indexable() > 0 {
            usize::MAX
        } else {
            0
        }
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<<I as Iterator>::Item> {
        let liter = self.iter.indexable();
        let lorig = self.orig.indexable();
        if lorig == 0 {
            None
        } else if index < liter {
            self.iter.idx(index)
        } else {
            self.orig.idx((index - liter) % lorig)
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
    flag: bool,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Chain<A, B> where
    A: Iterator,
    B: Iterator<Item = A::Item>
{
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        if self.flag {
            self.b.next()
        } else {
            match self.a.next() {
                Some(x) => return Some(x),
                _ => ()
            }
            self.flag = true;
            self.b.next()
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
        match self.b.next_back() {
            Some(x) => Some(x),
            None => self.a.next_back()
        }
    }
}

#[unstable(feature = "core", reason = "trait is experimental")]
impl<A, B> RandomAccessIterator for Chain<A, B> where
    A: RandomAccessIterator,
    B: RandomAccessIterator<Item = A::Item>,
{
    #[inline]
    fn indexable(&self) -> usize {
        let (a, b) = (self.a.indexable(), self.b.indexable());
        a.saturating_add(b)
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<A::Item> {
        let len = self.a.indexable();
        if index < len {
            self.a.idx(index)
        } else {
            self.b.idx(index - len)
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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<A, B> RandomAccessIterator for Zip<A, B> where
    A: RandomAccessIterator,
    B: RandomAccessIterator
{
    #[inline]
    fn indexable(&self) -> usize {
        cmp::min(self.a.indexable(), self.b.indexable())
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<(A::Item, B::Item)> {
        self.a.idx(index).and_then(|x| {
            self.b.idx(index).and_then(|y| {
                Some((x, y))
            })
        })
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
        self.iter.next().map(|a| (self.f)(a))
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
        self.iter.next_back().map(|a| (self.f)(a))
    }
}

#[unstable(feature = "core", reason = "trait is experimental")]
impl<B, I: RandomAccessIterator, F> RandomAccessIterator for Map<I, F> where
    F: FnMut(I::Item) -> B,
{
    #[inline]
    fn indexable(&self) -> usize {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<B> {
        self.iter.idx(index).map(|a| (self.f)(a))
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
    count: usize
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Enumerate<I> where I: Iterator {
    type Item = (usize, <I as Iterator>::Item);

    #[inline]
    fn next(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        self.iter.next().map(|a| {
            let ret = (self.count, a);
            self.count += 1;
            ret
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
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
            (self.count + len, a)
        })
    }
}

#[unstable(feature = "core", reason = "trait is experimental")]
impl<I> RandomAccessIterator for Enumerate<I> where I: RandomAccessIterator {
    #[inline]
    fn indexable(&self) -> usize {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<(usize, <I as Iterator>::Item)> {
        self.iter.idx(index).map(|a| (self.count + index, a))
    }
}

/// An iterator with a `peek()` that returns an optional reference to the next element.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Peekable<I: Iterator> {
    iter: I,
    peeked: Option<I::Item>,
}

impl<I: Iterator + Clone> Clone for Peekable<I> where I::Item: Clone {
    fn clone(&self) -> Peekable<I> {
        Peekable {
            iter: self.iter.clone(),
            peeked: self.peeked.clone(),
        }
    }
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
    /// Return a reference to the next element of the iterator with out
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

    /// Check whether peekable iterator is empty or not.
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
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        let mut next = self.iter.next();
        if self.n == 0 {
            next
        } else {
            let mut n = self.n;
            while n > 0 {
                n -= 1;
                match next {
                    Some(_) => {
                        next = self.iter.next();
                        continue
                    }
                    None => {
                        self.n = 0;
                        return None
                    }
                }
            }
            self.n = 0;
            next
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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<I> RandomAccessIterator for Skip<I> where I: RandomAccessIterator{
    #[inline]
    fn indexable(&self) -> usize {
        self.iter.indexable().saturating_sub(self.n)
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<<I as Iterator>::Item> {
        if index >= self.indexable() {
            None
        } else {
            self.iter.idx(index + self.n)
        }
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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<I> RandomAccessIterator for Take<I> where I: RandomAccessIterator{
    #[inline]
    fn indexable(&self) -> usize {
        cmp::min(self.iter.indexable(), self.n)
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<<I as Iterator>::Item> {
        if index >= self.n {
            None
        } else {
            self.iter.idx(index)
        }
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

    /// The current internal state to be passed to the closure next.
    #[unstable(feature = "core")]
    pub state: St,
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
            match self.iter.next().map(|x| (self.f)(x)) {
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
            match self.iter.next_back().map(|x| (self.f)(x)) {
                None => return self.frontiter.as_mut().and_then(|it| it.next_back()),
                next => self.backiter = next.map(IntoIterator::into_iter),
            }
        }
    }
}

/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
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
            match self.iter.next() {
                None => {
                    self.done = true;
                    None
                }
                x => x
            }
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
            match self.iter.next_back() {
                None => {
                    self.done = true;
                    None
                }
                x => x
            }
        }
    }
}

// Allow RandomAccessIterators to be fused without affecting random-access behavior
#[unstable(feature = "core", reason = "trait is experimental")]
impl<I> RandomAccessIterator for Fuse<I> where I: RandomAccessIterator {
    #[inline]
    fn indexable(&self) -> usize {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<<I as Iterator>::Item> {
        self.iter.idx(index)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Fuse<I> where I: ExactSizeIterator {}

impl<I> Fuse<I> {
    /// Resets the fuse such that the next call to .next() or .next_back() will
    /// call the underlying iterator again even if it previously returned None.
    #[inline]
    #[unstable(feature = "core", reason = "seems marginal")]
    pub fn reset_fuse(&mut self) {
        self.done = false
    }
}

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

#[unstable(feature = "core", reason = "trait is experimental")]
impl<I: RandomAccessIterator, F> RandomAccessIterator for Inspect<I, F>
    where F: FnMut(&I::Item),
{
    #[inline]
    fn indexable(&self) -> usize {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: usize) -> Option<I::Item> {
        let element = self.iter.idx(index);
        self.do_inspect(element)
    }
}

/// An iterator that passes mutable state to a closure and yields the result.
///
/// # Examples
///
/// An iterator that yields sequential Fibonacci numbers, and stops on overflow.
///
/// ```
/// # #![feature(core)]
/// use std::iter::Unfold;
/// use std::num::Int; // For `.checked_add()`
///
/// // This iterator will yield up to the last Fibonacci number before the max
/// // value of `u32`. You can simply change `u32` to `u64` in this line if
/// // you want higher values than that.
/// let mut fibonacci = Unfold::new((Some(0u32), Some(1u32)),
///                                 |&mut (ref mut x2, ref mut x1)| {
///     // Attempt to get the next Fibonacci number
///     // `x1` will be `None` if previously overflowed.
///     let next = match (*x2, *x1) {
///         (Some(x2), Some(x1)) => x2.checked_add(x1),
///         _ => None,
///     };
///
///     // Shift left: ret <- x2 <- x1 <- next
///     let ret = *x2;
///     *x2 = *x1;
///     *x1 = next;
///
///     ret
/// });
///
/// for i in fibonacci {
///     println!("{}", i);
/// }
/// ```
#[unstable(feature = "core")]
#[derive(Clone)]
pub struct Unfold<St, F> {
    f: F,
    /// Internal state that will be passed to the closure on the next iteration
    #[unstable(feature = "core")]
    pub state: St,
}

#[unstable(feature = "core")]
impl<A, St, F> Unfold<St, F> where F: FnMut(&mut St) -> Option<A> {
    /// Creates a new iterator with the specified closure as the "iterator
    /// function" and an initial state to eventually pass to the closure
    #[inline]
    pub fn new(initial_state: St, f: F) -> Unfold<St, F> {
        Unfold {
            f: f,
            state: initial_state
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, St, F> Iterator for Unfold<St, F> where F: FnMut(&mut St) -> Option<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        (self.f)(&mut self.state)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // no possible known bounds at this point
        (0, None)
    }
}

/// Objects that can be stepped over in both directions.
///
/// The `steps_between` function provides a way to efficiently compare
/// two `Step` objects.
#[unstable(feature = "step_trait",
           reason = "likely to be replaced by finer-grained traits")]
pub trait Step: PartialOrd {
    /// Steps `self` if possible.
    fn step(&self, by: &Self) -> Option<Self>;

    /// The number of steps between two step objects.
    ///
    /// `start` should always be less than `end`, so the result should never
    /// be negative.
    ///
    /// Return `None` if it is not possible to calculate steps_between
    /// without overflow.
    fn steps_between(start: &Self, end: &Self, by: &Self) -> Option<usize>;
}

macro_rules! step_impl {
    ($($t:ty)*) => ($(
        impl Step for $t {
            #[inline]
            fn step(&self, by: &$t) -> Option<$t> {
                (*self).checked_add(*by)
            }
            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t, by: &$t) -> Option<usize> {
                if *start <= *end {
                    Some(((*end - *start) / *by) as usize)
                } else {
                    Some(0)
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

step_impl!(usize u8 u16 u32 isize i8 i16 i32);
#[cfg(target_pointer_width = "64")]
step_impl!(u64 i64);
#[cfg(target_pointer_width = "32")]
step_impl_no_between!(u64 i64);

/// An adapter for stepping range iterators by a custom amount.
///
/// The resulting iterator handles overflow by stopping. The `A`
/// parameter is the type being iterated over, while `R` is the range
/// type (usually one of `std::ops::{Range, RangeFrom}`.
#[derive(Clone)]
#[unstable(feature = "step_by", reason = "recent addition")]
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
    #[unstable(feature = "step_by", reason = "recent addition")]
    pub fn step_by(self, by: A) -> StepBy<A, Self> {
        StepBy {
            step_by: by,
            range: self
        }
    }
}

#[allow(deprecated)]
impl<A: Step> ops::Range<A> {
    /// Creates an iterator with the same range, but stepping by the
    /// given amount at each iteration.
    ///
    /// The resulting iterator handles overflow by stopping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(step_by, core)]
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
    #[unstable(feature = "step_by", reason = "recent addition")]
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
#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
pub struct RangeInclusive<A> {
    range: ops::Range<A>,
    done: bool,
}

/// Return an iterator over the range [start, stop]
#[inline]
#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
pub fn range_inclusive<A>(start: A, stop: A) -> RangeInclusive<A>
    where A: Step + One + Clone
{
    RangeInclusive {
        range: start..stop,
        done: false,
    }
}

#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
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

#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
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
#[allow(deprecated)]
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
}

/// An iterator over the range [start, stop] by `step`. It handles overflow by stopping.
#[derive(Clone)]
#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
pub struct RangeStepInclusive<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
    done: bool,
}

/// Return an iterator over the range [start, stop] by `step`.
///
/// It handles overflow by stopping.
///
/// # Examples
///
/// ```
/// # #![feature(core)]
/// use std::iter::range_step_inclusive;
///
/// for i in range_step_inclusive(0, 10, 2) {
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
/// 10
/// ```
#[inline]
#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
#[allow(deprecated)]
pub fn range_step_inclusive<A: Int>(start: A, stop: A, step: A) -> RangeStepInclusive<A> {
    let rev = step < Int::zero();
    RangeStepInclusive {
        state: start,
        stop: stop,
        step: step,
        rev: rev,
        done: false,
    }
}

#[unstable(feature = "core",
           reason = "likely to be replaced by range notation and adapters")]
#[allow(deprecated)]
impl<A: Int> Iterator for RangeStepInclusive<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if !self.done && ((self.rev && self.state >= self.stop) ||
                          (!self.rev && self.state <= self.stop)) {
            let result = self.state;
            match self.state.checked_add(self.step) {
                Some(x) => self.state = x,
                None => self.done = true
            }
            Some(result)
        } else {
            None
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
#[allow(deprecated)]
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
    fn next(&mut self) -> Option<A> { self.idx(0) }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { (usize::MAX, None) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> DoubleEndedIterator for Repeat<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.idx(0) }
}

#[unstable(feature = "core", reason = "trait is experimental")]
impl<A: Clone> RandomAccessIterator for Repeat<A> {
    #[inline]
    fn indexable(&self) -> usize { usize::MAX }
    #[inline]
    fn idx(&mut self, _: usize) -> Option<A> { Some(self.element.clone()) }
}

type IterateState<T, F> = (F, Option<T>, bool);

/// An iterator that repeatedly applies a given function, starting
/// from a given seed value.
#[unstable(feature = "core")]
pub type Iterate<T, F> = Unfold<IterateState<T, F>, fn(&mut IterateState<T, F>) -> Option<T>>;

/// Create a new iterator that produces an infinite sequence of
/// repeated applications of the given function `f`.
#[unstable(feature = "core")]
pub fn iterate<T, F>(seed: T, f: F) -> Iterate<T, F> where
    T: Clone,
    F: FnMut(T) -> T,
{
    fn next<T, F>(st: &mut IterateState<T, F>) -> Option<T> where
        T: Clone,
        F: FnMut(T) -> T,
    {
        let &mut (ref mut f, ref mut val, ref mut first) = st;
        if *first {
            *first = false;
        } else if let Some(x) = val.take() {
            *val = Some((*f)(x))
        }
        val.clone()
    }

    // coerce to a fn pointer
    let next: fn(&mut IterateState<T,F>) -> Option<T> = next;

    Unfold::new((f, Some(seed), true), next)
}

/// Create a new iterator that endlessly repeats the element `elt`.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn repeat<T: Clone>(elt: T) -> Repeat<T> {
    Repeat{element: elt}
}

/// Functions for lexicographical ordering of sequences.
///
/// Lexicographical ordering through `<`, `<=`, `>=`, `>` requires
/// that the elements implement both `PartialEq` and `PartialOrd`.
///
/// If two sequences are equal up until the point where one ends,
/// the shorter sequence compares less.
#[unstable(feature = "core", reason = "needs review and revision")]
pub mod order {
    use cmp;
    use cmp::{Eq, Ord, PartialOrd, PartialEq};
    use cmp::Ordering::{Equal, Less, Greater};
    use option::Option;
    use option::Option::{Some, None};
    use super::Iterator;

    /// Compare `a` and `b` for equality using `Eq`
    pub fn equals<A, L, R>(mut a: L, mut b: R) -> bool where
        A: Eq,
        L: Iterator<Item=A>,
        R: Iterator<Item=A>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (None, _) | (_, None) => return false,
                (Some(x), Some(y)) => if x != y { return false },
            }
        }
    }

    /// Order `a` and `b` lexicographically using `Ord`
    pub fn cmp<A, L, R>(mut a: L, mut b: R) -> cmp::Ordering where
        A: Ord,
        L: Iterator<Item=A>,
        R: Iterator<Item=A>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return Equal,
                (None, _   ) => return Less,
                (_   , None) => return Greater,
                (Some(x), Some(y)) => match x.cmp(&y) {
                    Equal => (),
                    non_eq => return non_eq,
                },
            }
        }
    }

    /// Order `a` and `b` lexicographically using `PartialOrd`
    pub fn partial_cmp<L: Iterator, R: Iterator>(mut a: L, mut b: R) -> Option<cmp::Ordering> where
        L::Item: PartialOrd<R::Item>
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return Some(Equal),
                (None, _   ) => return Some(Less),
                (_   , None) => return Some(Greater),
                (Some(x), Some(y)) => match x.partial_cmp(&y) {
                    Some(Equal) => (),
                    non_eq => return non_eq,
                },
            }
        }
    }

    /// Compare `a` and `b` for equality (Using partial equality, `PartialEq`)
    pub fn eq<L: Iterator, R: Iterator>(mut a: L, mut b: R) -> bool where
        L::Item: PartialEq<R::Item>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (None, _) | (_, None) => return false,
                (Some(x), Some(y)) => if !x.eq(&y) { return false },
            }
        }
    }

    /// Compare `a` and `b` for nonequality (Using partial equality, `PartialEq`)
    pub fn ne<L: Iterator, R: Iterator>(mut a: L, mut b: R) -> bool where
        L::Item: PartialEq<R::Item>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return false,
                (None, _) | (_, None) => return true,
                (Some(x), Some(y)) => if x.ne(&y) { return true },
            }
        }
    }

    /// Return `a` < `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn lt<R: Iterator, L: Iterator>(mut a: L, mut b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return false,
                (None, _   ) => return true,
                (_   , None) => return false,
                (Some(x), Some(y)) => if x.ne(&y) { return x.lt(&y) },
            }
        }
    }

    /// Return `a` <= `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn le<L: Iterator, R: Iterator>(mut a: L, mut b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (None, _   ) => return true,
                (_   , None) => return false,
                (Some(x), Some(y)) => if x.ne(&y) { return x.le(&y) },
            }
        }
    }

    /// Return `a` > `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn gt<L: Iterator, R: Iterator>(mut a: L, mut b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return false,
                (None, _   ) => return false,
                (_   , None) => return true,
                (Some(x), Some(y)) => if x.ne(&y) { return x.gt(&y) },
            }
        }
    }

    /// Return `a` >= `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn ge<L: Iterator, R: Iterator>(mut a: L, mut b: R) -> bool where
        L::Item: PartialOrd<R::Item>,
    {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (None, _   ) => return false,
                (_   , None) => return true,
                (Some(x), Some(y)) => if x.ne(&y) { return x.ge(&y) },
            }
        }
    }
}
