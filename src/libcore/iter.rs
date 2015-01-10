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
//! This module defines Rust's core iteration trait. The `Iterator` trait has one
//! unimplemented method, `next`. All other methods are derived through default
//! methods to perform operations such as `zip`, `chain`, `enumerate`, and `fold`.
//!
//! The goal of this module is to unify iteration across all containers in Rust.
//! An iterator can be considered as a state machine which is used to track which
//! element will be yielded next.
//!
//! There are various extensions also defined in this module to assist with various
//! types of iteration, such as the `DoubleEndedIterator` for iterating in reverse,
//! the `FromIterator` trait for creating a container from an iterator, and much
//! more.
//!
//! ## Rust's `for` loop
//!
//! The special syntax used by rust's `for` loop is based around the `Iterator`
//! trait defined in this module. For loops can be viewed as a syntactical expansion
//! into a `loop`, for example, the `for` loop in this example is essentially
//! translated to the `loop` below.
//!
//! ```rust
//! let values = vec![1i, 2, 3];
//!
//! // "Syntactical sugar" taking advantage of an iterator
//! for &x in values.iter() {
//!     println!("{}", x);
//! }
//!
//! // Rough translation of the iteration without a `for` iterator.
//! let mut it = values.iter();
//! loop {
//!     match it.next() {
//!         Some(&x) => {
//!             println!("{}", x);
//!         }
//!         None => { break }
//!     }
//! }
//! ```
//!
//! This `for` loop syntax can be applied to any iterator over any type.

#![stable]

use self::MinMaxResult::*;

use clone::Clone;
use cmp;
use cmp::Ord;
use default::Default;
use mem;
use num::{ToPrimitive, Int};
use ops::{Add, Deref, FnMut};
use option::Option;
use option::Option::{Some, None};
use std::marker::Sized;
use uint;

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
#[stable]
pub trait Iterator {
    #[stable]
    type Item;

    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    #[stable]
    fn next(&mut self) -> Option<Self::Item>;

    /// Returns a lower and upper bound on the remaining length of the iterator.
    ///
    /// An upper bound of `None` means either there is no known upper bound, or the upper bound
    /// does not fit within a `uint`.
    #[inline]
    #[stable]
    fn size_hint(&self) -> (uint, Option<uint>) { (0, None) }
}

/// Conversion from an `Iterator`
#[stable]
pub trait FromIterator<A> {
    /// Build a container with elements from an external iterator.
    fn from_iter<T: Iterator<Item=A>>(iterator: T) -> Self;
}

/// A type growable from an `Iterator` implementation
#[stable]
pub trait Extend<A> {
    /// Extend a container with the elements yielded by an arbitrary iterator
    fn extend<T: Iterator<Item=A>>(&mut self, iterator: T);
}

/// An extension trait providing numerous methods applicable to all iterators.
#[stable]
pub trait IteratorExt: Iterator + Sized {
    /// Counts the number of elements in this iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.count() == 5);
    /// ```
    #[inline]
    #[stable]
    fn count(self) -> uint {
        self.fold(0, |cnt, _x| cnt + 1)
    }

    /// Loops through the entire iterator, returning the last element of the
    /// iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().last().unwrap() == &5);
    /// ```
    #[inline]
    #[stable]
    fn last(mut self) -> Option<Self::Item> {
        let mut last = None;
        for x in self { last = Some(x); }
        last
    }

    /// Loops through `n` iterations, returning the `n`th element of the
    /// iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.nth(2).unwrap() == &3);
    /// assert!(it.nth(2) == None);
    /// ```
    #[inline]
    #[stable]
    fn nth(&mut self, mut n: uint) -> Option<Self::Item> {
        for x in *self {
            if n == 0 { return Some(x) }
            n -= 1;
        }
        None
    }

    /// Chain this iterator with another, returning a new iterator that will
    /// finish iterating over the current iterator, and then iterate
    /// over the other specified iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [0i];
    /// let b = [1i];
    /// let mut it = a.iter().chain(b.iter());
    /// assert_eq!(it.next().unwrap(), &0);
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn chain<U>(self, other: U) -> Chain<Self, U> where
        U: Iterator<Item=Self::Item>,
    {
        Chain{a: self, b: other, flag: false}
    }

    /// Creates an iterator that iterates over both this and the specified
    /// iterators simultaneously, yielding the two elements as pairs. When
    /// either iterator returns None, all further invocations of next() will
    /// return None.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [0i];
    /// let b = [1i];
    /// let mut it = a.iter().zip(b.iter());
    /// let (x0, x1) = (0i, 1i);
    /// assert_eq!(it.next().unwrap(), (&x0, &x1));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn zip<B, U>(self, other: U) -> Zip<Self, U> where
        U: Iterator<Item=B>,
    {
        Zip{a: self, b: other}
    }

    /// Creates a new iterator that will apply the specified function to each
    /// element returned by the first, yielding the mapped element instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2];
    /// let mut it = a.iter().map(|&x| 2 * x);
    /// assert_eq!(it.next().unwrap(), 2);
    /// assert_eq!(it.next().unwrap(), 4);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn map<B, F>(self, f: F) -> Map<Self::Item, B, Self, F> where
        F: FnMut(Self::Item) -> B,
    {
        Map{iter: self, f: f}
    }

    /// Creates an iterator that applies the predicate to each element returned
    /// by this iterator. Only elements that have the predicate evaluate to
    /// `true` will be yielded.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2];
    /// let mut it = a.iter().filter(|&x| *x > 1);
    /// assert_eq!(it.next().unwrap(), &2);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn filter<P>(self, predicate: P) -> Filter<Self::Item, Self, P> where
        P: FnMut(&Self::Item) -> bool,
    {
        Filter{iter: self, predicate: predicate}
    }

    /// Creates an iterator that both filters and maps elements.
    /// If the specified function returns None, the element is skipped.
    /// Otherwise the option is unwrapped and the new value is yielded.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2];
    /// let mut it = a.iter().filter_map(|&x| if x > 1 {Some(2 * x)} else {None});
    /// assert_eq!(it.next().unwrap(), 4);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self::Item, B, Self, F> where
        F: FnMut(Self::Item) -> Option<B>,
    {
        FilterMap { iter: self, f: f }
    }

    /// Creates an iterator that yields a pair of the value returned by this
    /// iterator plus the current index of iteration.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [100i, 200];
    /// let mut it = a.iter().enumerate();
    /// let (x100, x200) = (100i, 200i);
    /// assert_eq!(it.next().unwrap(), (0, &x100));
    /// assert_eq!(it.next().unwrap(), (1, &x200));
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn enumerate(self) -> Enumerate<Self> {
        Enumerate{iter: self, count: 0}
    }

    /// Creates an iterator that has a `.peek()` method
    /// that returns an optional reference to the next element.
    ///
    /// # Example
    ///
    /// ```rust
    /// let xs = [100i, 200, 300];
    /// let mut it = xs.iter().map(|x| *x).peekable();
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
    #[stable]
    fn peekable(self) -> Peekable<Self::Item, Self> {
        Peekable{iter: self, peeked: None}
    }

    /// Creates an iterator that invokes the predicate on elements
    /// until it returns false. Once the predicate returns false, that
    /// element and all further elements are yielded.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 2, 1];
    /// let mut it = a.iter().skip_while(|&a| *a < 3);
    /// assert_eq!(it.next().unwrap(), &3);
    /// assert_eq!(it.next().unwrap(), &2);
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn skip_while<P>(self, predicate: P) -> SkipWhile<Self::Item, Self, P> where
        P: FnMut(&Self::Item) -> bool,
    {
        SkipWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator that yields elements so long as the predicate
    /// returns true. After the predicate returns false for the first time, no
    /// further elements will be yielded.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 2, 1];
    /// let mut it = a.iter().take_while(|&a| *a < 3);
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert_eq!(it.next().unwrap(), &2);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn take_while<P>(self, predicate: P) -> TakeWhile<Self::Item, Self, P> where
        P: FnMut(&Self::Item) -> bool,
    {
        TakeWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator that skips the first `n` elements of this iterator,
    /// and then yields all further items.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter().skip(3);
    /// assert_eq!(it.next().unwrap(), &4);
    /// assert_eq!(it.next().unwrap(), &5);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn skip(self, n: uint) -> Skip<Self> {
        Skip{iter: self, n: n}
    }

    /// Creates an iterator that yields the first `n` elements of this
    /// iterator, and then will always return None.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter().take(3);
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert_eq!(it.next().unwrap(), &2);
    /// assert_eq!(it.next().unwrap(), &3);
    /// assert!(it.next().is_none());
    /// ```
    #[inline]
    #[stable]
    fn take(self, n: uint) -> Take<Self> {
        Take{iter: self, n: n}
    }

    /// Creates a new iterator that behaves in a similar fashion to fold.
    /// There is a state which is passed between each iteration and can be
    /// mutated as necessary. The yielded values from the closure are yielded
    /// from the Scan instance when not None.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
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
    #[stable]
    fn scan<St, B, F>(
        self,
        initial_state: St,
        f: F,
    ) -> Scan<Self::Item, B, Self, St, F> where
        F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        Scan{iter: self, f: f, state: initial_state}
    }

    /// Creates an iterator that maps each element to an iterator,
    /// and yields the elements of the produced iterators
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::count;
    ///
    /// let xs = [2u, 3];
    /// let ys = [0u, 1, 0, 1, 2];
    /// let mut it = xs.iter().flat_map(|&x| count(0u, 1).take(x));
    /// // Check that `it` has the same elements as `ys`
    /// let mut i = 0;
    /// for x in it {
    ///     assert_eq!(x, ys[i]);
    ///     i += 1;
    /// }
    /// ```
    #[inline]
    #[stable]
    fn flat_map<B, U, F>(self, f: F) -> FlatMap<Self::Item, B, Self, U, F> where
        U: Iterator<Item=B>,
        F: FnMut(Self::Item) -> U,
    {
        FlatMap{iter: self, f: f, frontiter: None, backiter: None }
    }

    /// Creates an iterator that yields `None` forever after the underlying
    /// iterator yields `None`. Random-access iterator behavior is not
    /// affected, only single and double-ended iterator behavior.
    ///
    /// # Example
    ///
    /// ```rust
    /// fn process<U: Iterator<Item=int>>(it: U) -> int {
    ///     let mut it = it.fuse();
    ///     let mut sum = 0;
    ///     for x in it {
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
    /// let x = vec![1i,2,3,7,8,9];
    /// assert_eq!(process(x.into_iter()), 6);
    /// let x = vec![1i,2,3];
    /// assert_eq!(process(x.into_iter()), 1006);
    /// ```
    #[inline]
    #[stable]
    fn fuse(self) -> Fuse<Self> {
        Fuse{iter: self, done: false}
    }

    /// Creates an iterator that calls a function with a reference to each
    /// element before yielding it. This is often useful for debugging an
    /// iterator pipeline.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::AdditiveIterator;
    ///
    /// let xs = [1u, 4, 2, 3, 8, 9, 6];
    /// let sum = xs.iter()
    ///             .map(|&x| x)
    ///             .inspect(|&x| println!("filtering {}", x))
    ///             .filter(|&x| x % 2 == 0)
    ///             .inspect(|&x| println!("{} made it through", x))
    ///             .sum();
    /// println!("{}", sum);
    /// ```
    #[inline]
    #[stable]
    fn inspect<F>(self, f: F) -> Inspect<Self::Item, Self, F> where
        F: FnMut(&Self::Item),
    {
        Inspect{iter: self, f: f}
    }

    /// Creates a wrapper around a mutable reference to the iterator.
    ///
    /// This is useful to allow applying iterator adaptors while still
    /// retaining ownership of the original iterator value.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut xs = range(0u, 10);
    /// // sum the first five values
    /// let partial_sum = xs.by_ref().take(5).fold(0, |a, b| a + b);
    /// assert!(partial_sum == 10);
    /// // xs.next() is now `5`
    /// assert!(xs.next() == Some(5));
    /// ```
    #[stable]
    fn by_ref<'r>(&'r mut self) -> ByRef<'r, Self> {
        ByRef{iter: self}
    }

    /// Loops through the entire iterator, collecting all of the elements into
    /// a container implementing `FromIterator`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let b: Vec<int> = a.iter().map(|&x| x).collect();
    /// assert!(a.as_slice() == b.as_slice());
    /// ```
    #[inline]
    #[stable]
    fn collect<B: FromIterator<Self::Item>>(self) -> B {
        FromIterator::from_iter(self)
    }

    /// Loops through the entire iterator, collecting all of the elements into
    /// one of two containers, depending on a predicate. The elements of the
    /// first container satisfy the predicate, while the elements of the second
    /// do not.
    ///
    /// ```
    /// let vec = vec![1i, 2i, 3i, 4i];
    /// let (even, odd): (Vec<int>, Vec<int>) = vec.into_iter().partition(|&n| n % 2 == 0);
    /// assert_eq!(even, vec![2, 4]);
    /// assert_eq!(odd, vec![1, 3]);
    /// ```
    #[unstable = "recently added as part of collections reform"]
    fn partition<B, F>(mut self, mut f: F) -> (B, B) where
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
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().fold(0, |a, &b| a + b) == 15);
    /// ```
    #[inline]
    #[stable]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        for x in self {
            accum = f(accum, x);
        }
        accum
    }

    /// Tests whether the predicate holds true for all elements in the iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().all(|x| *x > 0));
    /// assert!(!a.iter().all(|x| *x > 2));
    /// ```
    #[inline]
    #[stable]
    fn all<F>(mut self, mut f: F) -> bool where F: FnMut(Self::Item) -> bool {
        for x in self { if !f(x) { return false; } }
        true
    }

    /// Tests whether any element of an iterator satisfies the specified
    /// predicate.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.any(|x| *x == 3));
    /// assert!(!it.any(|x| *x == 3));
    /// ```
    #[inline]
    #[stable]
    fn any<F>(&mut self, mut f: F) -> bool where F: FnMut(Self::Item) -> bool {
        for x in *self { if f(x) { return true; } }
        false
    }

    /// Returns the first element satisfying the specified predicate.
    ///
    /// Does not consume the iterator past the first found element.
    #[inline]
    #[stable]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item> where
        P: FnMut(&Self::Item) -> bool,
    {
        for x in *self {
            if predicate(&x) { return Some(x) }
        }
        None
    }

    /// Return the index of the first element satisfying the specified predicate
    #[inline]
    #[stable]
    fn position<P>(&mut self, mut predicate: P) -> Option<uint> where
        P: FnMut(Self::Item) -> bool,
    {
        let mut i = 0;
        for x in *self {
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
    #[inline]
    #[stable]
    fn rposition<P>(&mut self, mut predicate: P) -> Option<uint> where
        P: FnMut(Self::Item) -> bool,
        Self: ExactSizeIterator + DoubleEndedIterator
    {
        let len = self.len();
        for i in range(0, len).rev() {
            if predicate(self.next_back().expect("rposition: incorrect ExactSizeIterator")) {
                return Some(i);
            }
        }
        None
    }

    /// Consumes the entire iterator to return the maximum element.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().max().unwrap() == &5);
    /// ```
    #[inline]
    #[stable]
    fn max(self) -> Option<Self::Item> where Self::Item: Ord
    {
        self.fold(None, |max, x| {
            match max {
                None    => Some(x),
                Some(y) => Some(cmp::max(x, y))
            }
        })
    }

    /// Consumes the entire iterator to return the minimum element.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().min().unwrap() == &1);
    /// ```
    #[inline]
    #[stable]
    fn min(self) -> Option<Self::Item> where Self::Item: Ord
    {
        self.fold(None, |min, x| {
            match min {
                None    => Some(x),
                Some(y) => Some(cmp::min(x, y))
            }
        })
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
    /// and so is faster than calling `min` and `max` separately which does `2 * n` comparisons.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::MinMaxResult::{NoElements, OneElement, MinMax};
    ///
    /// let v: [int; 0] = [];
    /// assert_eq!(v.iter().min_max(), NoElements);
    ///
    /// let v = [1i];
    /// assert!(v.iter().min_max() == OneElement(&1));
    ///
    /// let v = [1i, 2, 3, 4, 5];
    /// assert!(v.iter().min_max() == MinMax(&1, &5));
    ///
    /// let v = [1i, 2, 3, 4, 5, 6];
    /// assert!(v.iter().min_max() == MinMax(&1, &6));
    ///
    /// let v = [1i, 1, 1, 1];
    /// assert!(v.iter().min_max() == MinMax(&1, &1));
    /// ```
    #[unstable = "return type may change"]
    fn min_max(mut self) -> MinMaxResult<Self::Item> where Self::Item: Ord
    {
        let (mut min, mut max) = match self.next() {
            None => return NoElements,
            Some(x) => {
                match self.next() {
                    None => return OneElement(x),
                    Some(y) => if x < y {(x, y)} else {(y,x)}
                }
            }
        };

        loop {
            // `first` and `second` are the two next elements we want to look at.
            // We first compare `first` and `second` (#1). The smaller one is then compared to
            // current minimum (#2). The larger one is compared to current maximum (#3). This
            // way we do 3 comparisons for 2 elements.
            let first = match self.next() {
                None => break,
                Some(x) => x
            };
            let second = match self.next() {
                None => {
                    if first < min {
                        min = first;
                    } else if first > max {
                        max = first;
                    }
                    break;
                }
                Some(x) => x
            };
            if first < second {
                if first < min {min = first;}
                if max < second {max = second;}
            } else {
                if second < min {min = second;}
                if max < first {max = first;}
            }
        }

        MinMax(min, max)
    }

    /// Return the element that gives the maximum value from the
    /// specified function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use core::num::SignedInt;
    ///
    /// let xs = [-3i, 0, 1, 5, -10];
    /// assert_eq!(*xs.iter().max_by(|x| x.abs()).unwrap(), -10);
    /// ```
    #[inline]
    #[unstable = "may want to produce an Ordering directly; see #15311"]
    fn max_by<B: Ord, F>(self, mut f: F) -> Option<Self::Item> where
        F: FnMut(&Self::Item) -> B,
    {
        self.fold(None, |max: Option<(Self::Item, B)>, x| {
            let x_val = f(&x);
            match max {
                None             => Some((x, x_val)),
                Some((y, y_val)) => if x_val > y_val {
                    Some((x, x_val))
                } else {
                    Some((y, y_val))
                }
            }
        }).map(|(x, _)| x)
    }

    /// Return the element that gives the minimum value from the
    /// specified function.
    ///
    /// # Example
    ///
    /// ```rust
    /// use core::num::SignedInt;
    ///
    /// let xs = [-3i, 0, 1, 5, -10];
    /// assert_eq!(*xs.iter().min_by(|x| x.abs()).unwrap(), 0);
    /// ```
    #[inline]
    #[unstable = "may want to produce an Ordering directly; see #15311"]
    fn min_by<B: Ord, F>(self, mut f: F) -> Option<Self::Item> where
        F: FnMut(&Self::Item) -> B,
    {
        self.fold(None, |min: Option<(Self::Item, B)>, x| {
            let x_val = f(&x);
            match min {
                None             => Some((x, x_val)),
                Some((y, y_val)) => if x_val < y_val {
                    Some((x, x_val))
                } else {
                    Some((y, y_val))
                }
            }
        }).map(|(x, _)| x)
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
    /// `uint::MAX` elements of the original iterator.
    #[inline]
    #[stable]
    fn rev(self) -> Rev<Self> {
        Rev{iter: self}
    }

    /// Converts an iterator of pairs into a pair of containers.
    ///
    /// Loops through the entire iterator, collecting the first component of
    /// each item into one new container, and the second component into another.
    #[unstable = "recent addition"]
    fn unzip<A, B, FromA, FromB>(mut self) -> (FromA, FromB) where
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
        Self: Iterator<Item=(A, B)>,
    {
        struct SizeHint<A>(uint, Option<uint>);
        impl<A> Iterator for SizeHint<A> {
            type Item = A;

            fn next(&mut self) -> Option<A> { None }
            fn size_hint(&self) -> (uint, Option<uint>) {
                (self.0, self.1)
            }
        }

        let (lo, hi) = self.size_hint();
        let mut ts: FromA = Default::default();
        let mut us: FromB = Default::default();

        ts.extend(SizeHint(lo, hi));
        us.extend(SizeHint(lo, hi));

        for (t, u) in self {
            ts.extend(Some(t).into_iter());
            us.extend(Some(u).into_iter());
        }

        (ts, us)
    }

    /// Creates an iterator that clones the elements it yields. Useful for converting an
    /// Iterator<&T> to an Iterator<T>.
    #[unstable = "recent addition"]
    fn cloned<T, D>(self) -> Cloned<Self> where
        Self: Iterator<Item=D>,
        D: Deref<Target=T>,
        T: Clone,
    {
        Cloned { it: self }
    }

    /// Repeats an iterator endlessly
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::count;
    ///
    /// let a = count(1i,1i).take(1);
    /// let mut cy = a.cycle();
    /// assert_eq!(cy.next(), Some(1));
    /// assert_eq!(cy.next(), Some(1));
    /// ```
    #[stable]
    #[inline]
    fn cycle(self) -> Cycle<Self> where Self: Clone {
        Cycle{orig: self.clone(), iter: self}
    }

    /// Use an iterator to reverse a container in place.
    #[unstable = "uncertain about placement or widespread use"]
    fn reverse_in_place<'a, T: 'a>(&mut self) where
        Self: Iterator<Item=&'a mut T> + DoubleEndedIterator
    {
        loop {
            match (self.next(), self.next_back()) {
                (Some(x), Some(y)) => mem::swap(x, y),
                _ => break
            }
        }
    }
}

#[stable]
impl<I> IteratorExt for I where I: Iterator {}

/// A range iterator able to yield elements from both ends
///
/// A `DoubleEndedIterator` can be thought of as a deque in that `next()` and `next_back()` exhaust
/// elements from the *same* range, and do not work independently of each other.
#[stable]
pub trait DoubleEndedIterator: Iterator {
    /// Yield an element from the end of the range, returning `None` if the range is empty.
    fn next_back(&mut self) -> Option<Self::Item>;
}

/// An object implementing random access indexing by `uint`
///
/// A `RandomAccessIterator` should be either infinite or a `DoubleEndedIterator`.
/// Calling `next()` or `next_back()` on a `RandomAccessIterator`
/// reduces the indexable range accordingly. That is, `it.idx(1)` will become `it.idx(0)`
/// after `it.next()` is called.
#[unstable = "not widely used, may be better decomposed into Index and ExactSizeIterator"]
pub trait RandomAccessIterator: Iterator {
    /// Return the number of indexable elements. At most `std::uint::MAX`
    /// elements are indexable, even if the iterator represents a longer range.
    fn indexable(&self) -> uint;

    /// Return an element at an index, or `None` if the index is out of bounds
    fn idx(&mut self, index: uint) -> Option<Self::Item>;
}

/// An iterator that knows its exact length
///
/// This trait is a helper for iterators like the vector iterator, so that
/// it can support double-ended enumeration.
///
/// `Iterator::size_hint` *must* return the exact size of the iterator.
/// Note that the size must fit in `uint`.
#[stable]
pub trait ExactSizeIterator: Iterator {
    #[inline]
    /// Return the exact length of the iterator.
    fn len(&self) -> uint {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }
}

// All adaptors that preserve the size of the wrapped iterator are fine
// Adaptors that may overflow in `size_hint` are not, i.e. `Chain`.
#[stable]
impl<I> ExactSizeIterator for Enumerate<I> where I: ExactSizeIterator {}
#[stable]
impl<A, I, F> ExactSizeIterator for Inspect<A, I, F> where
    I: ExactSizeIterator<Item=A>,
    F: FnMut(&A),
{}
#[stable]
impl<I> ExactSizeIterator for Rev<I> where I: ExactSizeIterator + DoubleEndedIterator {}
#[stable]
impl<A, B, I, F> ExactSizeIterator for Map<A, B, I, F> where
    I: ExactSizeIterator<Item=A>,
    F: FnMut(A) -> B,
{}
#[stable]
impl<A, B> ExactSizeIterator for Zip<A, B> where A: ExactSizeIterator, B: ExactSizeIterator {}

/// An double-ended iterator with the direction inverted
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Rev<T> {
    iter: T
}

#[stable]
impl<I> Iterator for Rev<I> where I: DoubleEndedIterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> { self.iter.next_back() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

#[stable]
impl<I> DoubleEndedIterator for Rev<I> where I: DoubleEndedIterator {
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> { self.iter.next() }
}

#[unstable = "trait is experimental"]
impl<I> RandomAccessIterator for Rev<I> where I: DoubleEndedIterator + RandomAccessIterator {
    #[inline]
    fn indexable(&self) -> uint { self.iter.indexable() }
    #[inline]
    fn idx(&mut self, index: uint) -> Option<<I as Iterator>::Item> {
        let amt = self.indexable();
        self.iter.idx(amt - index - 1)
    }
}

/// A mutable reference to an iterator
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct ByRef<'a, I:'a> {
    iter: &'a mut I,
}

#[stable]
impl<'a, I> Iterator for ByRef<'a, I> where I: 'a + Iterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> { self.iter.next() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

#[stable]
impl<'a, I> DoubleEndedIterator for ByRef<'a, I> where I: 'a + DoubleEndedIterator {
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> { self.iter.next_back() }
}

/// A trait for iterators over elements which can be added together
#[unstable = "needs to be re-evaluated as part of numerics reform"]
pub trait AdditiveIterator<A> {
    /// Iterates over the entire iterator, summing up all the elements
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::AdditiveIterator;
    ///
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter().map(|&x| x);
    /// assert!(it.sum() == 15);
    /// ```
    fn sum(self) -> A;
}

macro_rules! impl_additive {
    ($A:ty, $init:expr) => {
        #[unstable = "trait is experimental"]
        impl<T: Iterator<Item=$A>> AdditiveIterator<$A> for T {
            #[inline]
            fn sum(self) -> $A {
                self.fold($init, |acc, x| acc + x)
            }
        }
    };
}
impl_additive! { i8,   0 }
impl_additive! { i16,  0 }
impl_additive! { i32,  0 }
impl_additive! { i64,  0 }
impl_additive! { int,  0 }
impl_additive! { u8,   0 }
impl_additive! { u16,  0 }
impl_additive! { u32,  0 }
impl_additive! { u64,  0 }
impl_additive! { uint, 0 }
impl_additive! { f32,  0.0 }
impl_additive! { f64,  0.0 }

/// A trait for iterators over elements which can be multiplied together.
#[unstable = "needs to be re-evaluated as part of numerics reform"]
pub trait MultiplicativeIterator<A> {
    /// Iterates over the entire iterator, multiplying all the elements
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::{count, MultiplicativeIterator};
    ///
    /// fn factorial(n: uint) -> uint {
    ///     count(1u, 1).take_while(|&i| i <= n).product()
    /// }
    /// assert!(factorial(0) == 1);
    /// assert!(factorial(1) == 1);
    /// assert!(factorial(5) == 120);
    /// ```
    fn product(self) -> A;
}

macro_rules! impl_multiplicative {
    ($A:ty, $init:expr) => {
        #[unstable = "trait is experimental"]
        impl<T: Iterator<Item=$A>> MultiplicativeIterator<$A> for T {
            #[inline]
            fn product(self) -> $A {
                self.fold($init, |acc, x| acc * x)
            }
        }
    };
}
impl_multiplicative! { i8,   1 }
impl_multiplicative! { i16,  1 }
impl_multiplicative! { i32,  1 }
impl_multiplicative! { i64,  1 }
impl_multiplicative! { int,  1 }
impl_multiplicative! { u8,   1 }
impl_multiplicative! { u16,  1 }
impl_multiplicative! { u32,  1 }
impl_multiplicative! { u64,  1 }
impl_multiplicative! { uint, 1 }
impl_multiplicative! { f32,  1.0 }
impl_multiplicative! { f64,  1.0 }

/// `MinMaxResult` is an enum returned by `min_max`. See `IteratorOrdExt::min_max` for more detail.
#[derive(Clone, PartialEq, Show)]
#[unstable = "unclear whether such a fine-grained result is widely useful"]
pub enum MinMaxResult<T> {
    /// Empty iterator
    NoElements,

    /// Iterator with one element, so the minimum and maximum are the same
    OneElement(T),

    /// More than one element in the iterator, the first element is not larger than the second
    MinMax(T, T)
}

impl<T: Clone> MinMaxResult<T> {
    /// `into_option` creates an `Option` of type `(T,T)`. The returned `Option` has variant
    /// `None` if and only if the `MinMaxResult` has variant `NoElements`. Otherwise variant
    /// `Some(x,y)` is returned where `x <= y`. If `MinMaxResult` has variant `OneElement(x)`,
    /// performing this operation will make one clone of `x`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::MinMaxResult::{self, NoElements, OneElement, MinMax};
    ///
    /// let r: MinMaxResult<int> = NoElements;
    /// assert_eq!(r.into_option(), None);
    ///
    /// let r = OneElement(1i);
    /// assert_eq!(r.into_option(), Some((1,1)));
    ///
    /// let r = MinMax(1i,2i);
    /// assert_eq!(r.into_option(), Some((1,2)));
    /// ```
    #[unstable = "type is unstable"]
    pub fn into_option(self) -> Option<(T,T)> {
        match self {
            NoElements => None,
            OneElement(x) => Some((x.clone(), x)),
            MinMax(x, y) => Some((x, y))
        }
    }
}

/// An iterator that clones the elements of an underlying iterator
#[unstable = "recent addition"]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct Cloned<I> {
    it: I,
}

#[stable]
impl<T, D, I> Iterator for Cloned<I> where
    T: Clone,
    D: Deref<Target=T>,
    I: Iterator<Item=D>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.it.next().cloned()
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.it.size_hint()
    }
}

#[stable]
impl<T, D, I> DoubleEndedIterator for Cloned<I> where
    T: Clone,
    D: Deref<Target=T>,
    I: DoubleEndedIterator<Item=D>,
{
    fn next_back(&mut self) -> Option<T> {
        self.it.next_back().cloned()
    }
}

#[stable]
impl<T, D, I> ExactSizeIterator for Cloned<I> where
    T: Clone,
    D: Deref<Target=T>,
    I: ExactSizeIterator<Item=D>,
{}

/// An iterator that repeats endlessly
#[derive(Clone, Copy)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Cycle<I> {
    orig: I,
    iter: I,
}

#[stable]
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
    fn size_hint(&self) -> (uint, Option<uint>) {
        // the cycle iterator is either empty or infinite
        match self.orig.size_hint() {
            sz @ (0, Some(0)) => sz,
            (0, _) => (0, None),
            _ => (uint::MAX, None)
        }
    }
}

#[unstable = "trait is experimental"]
impl<I> RandomAccessIterator for Cycle<I> where
    I: Clone + RandomAccessIterator,
{
    #[inline]
    fn indexable(&self) -> uint {
        if self.orig.indexable() > 0 {
            uint::MAX
        } else {
            0
        }
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<<I as Iterator>::Item> {
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
#[stable]
pub struct Chain<A, B> {
    a: A,
    b: B,
    flag: bool,
}

#[stable]
impl<T, A, B> Iterator for Chain<A, B> where A: Iterator<Item=T>, B: Iterator<Item=T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
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
    fn size_hint(&self) -> (uint, Option<uint>) {
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

#[stable]
impl<T, A, B> DoubleEndedIterator for Chain<A, B> where
    A: DoubleEndedIterator<Item=T>,
    B: DoubleEndedIterator<Item=T>,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        match self.b.next_back() {
            Some(x) => Some(x),
            None => self.a.next_back()
        }
    }
}

#[unstable = "trait is experimental"]
impl<T, A, B> RandomAccessIterator for Chain<A, B> where
    A: RandomAccessIterator<Item=T>,
    B: RandomAccessIterator<Item=T>,
{
    #[inline]
    fn indexable(&self) -> uint {
        let (a, b) = (self.a.indexable(), self.b.indexable());
        a.saturating_add(b)
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<T> {
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
#[stable]
pub struct Zip<A, B> {
    a: A,
    b: B
}

#[stable]
impl<T, U, A, B> Iterator for Zip<A, B> where
    A: Iterator<Item = T>,
    B: Iterator<Item = U>,
{
    type Item = (T, U);

    #[inline]
    fn next(&mut self) -> Option<(T, U)> {
        match self.a.next() {
            None => None,
            Some(x) => match self.b.next() {
                None => None,
                Some(y) => Some((x, y))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
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

#[stable]
impl<T, U, A, B> DoubleEndedIterator for Zip<A, B> where
    A: DoubleEndedIterator + ExactSizeIterator<Item=T>,
    B: DoubleEndedIterator + ExactSizeIterator<Item=U>,
{
    #[inline]
    fn next_back(&mut self) -> Option<(T, U)> {
        let a_sz = self.a.len();
        let b_sz = self.b.len();
        if a_sz != b_sz {
            // Adjust a, b to equal length
            if a_sz > b_sz {
                for _ in range(0, a_sz - b_sz) { self.a.next_back(); }
            } else {
                for _ in range(0, b_sz - a_sz) { self.b.next_back(); }
            }
        }
        match (self.a.next_back(), self.b.next_back()) {
            (Some(x), Some(y)) => Some((x, y)),
            (None, None) => None,
            _ => unreachable!(),
        }
    }
}

#[unstable = "trait is experimental"]
impl<T, U, A, B> RandomAccessIterator for Zip<A, B> where
    A: RandomAccessIterator<Item=T>,
    B: RandomAccessIterator<Item=U>,
{
    #[inline]
    fn indexable(&self) -> uint {
        cmp::min(self.a.indexable(), self.b.indexable())
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<(T, U)> {
        match self.a.idx(index) {
            None => None,
            Some(x) => match self.b.idx(index) {
                None => None,
                Some(y) => Some((x, y))
            }
        }
    }
}

/// An iterator that maps the values of `iter` with `f`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Map<A, B, I: Iterator<Item=A>, F: FnMut(A) -> B> {
    iter: I,
    f: F,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, B, I, F> Clone for Map<A, B, I, F> where
    I: Clone + Iterator<Item=A>,
    F: Clone + FnMut(A) -> B,
{
    fn clone(&self) -> Map<A, B, I, F> {
        Map {
            iter: self.iter.clone(),
            f: self.f.clone(),
        }
    }
}

impl<A, B, I, F> Map<A, B, I, F> where I: Iterator<Item=A>, F: FnMut(A) -> B {
    #[inline]
    fn do_map(&mut self, elt: Option<A>) -> Option<B> {
        match elt {
            Some(a) => Some((self.f)(a)),
            _ => None
        }
    }
}

#[stable]
impl<A, B, I, F> Iterator for Map<A, B, I, F> where I: Iterator<Item=A>, F: FnMut(A) -> B {
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        let next = self.iter.next();
        self.do_map(next)
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

#[stable]
impl<A, B, I, F> DoubleEndedIterator for Map<A, B, I, F> where
    I: DoubleEndedIterator<Item=A>,
    F: FnMut(A) -> B,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        let next = self.iter.next_back();
        self.do_map(next)
    }
}

#[unstable = "trait is experimental"]
impl<A, B, I, F> RandomAccessIterator for Map<A, B, I, F> where
    I: RandomAccessIterator<Item=A>,
    F: FnMut(A) -> B,
{
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<B> {
        let elt = self.iter.idx(index);
        self.do_map(elt)
    }
}

/// An iterator that filters the elements of `iter` with `predicate`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Filter<A, I, P> where I: Iterator<Item=A>, P: FnMut(&A) -> bool {
    iter: I,
    predicate: P,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, I, P> Clone for Filter<A, I, P> where
    I: Clone + Iterator<Item=A>,
    P: Clone + FnMut(&A) -> bool,
{
    fn clone(&self) -> Filter<A, I, P> {
        Filter {
            iter: self.iter.clone(),
            predicate: self.predicate.clone(),
        }
    }
}

#[stable]
impl<A, I, P> Iterator for Filter<A, I, P> where I: Iterator<Item=A>, P: FnMut(&A) -> bool {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        for x in self.iter {
            if (self.predicate)(&x) {
                return Some(x);
            } else {
                continue
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

#[stable]
impl<A, I, P> DoubleEndedIterator for Filter<A, I, P> where
    I: DoubleEndedIterator<Item=A>,
    P: FnMut(&A) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<A> {
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
#[stable]
pub struct FilterMap<A, B, I, F> where I: Iterator<Item=A>, F: FnMut(A) -> Option<B> {
    iter: I,
    f: F,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, B, I, F> Clone for FilterMap<A, B, I, F> where
    I: Clone + Iterator<Item=A>,
    F: Clone + FnMut(A) -> Option<B>,
{
    fn clone(&self) -> FilterMap<A, B, I, F> {
        FilterMap {
            iter: self.iter.clone(),
            f: self.f.clone(),
        }
    }
}

#[stable]
impl<A, B, I, F> Iterator for FilterMap<A, B, I, F> where
    I: Iterator<Item=A>,
    F: FnMut(A) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        for x in self.iter {
            match (self.f)(x) {
                Some(y) => return Some(y),
                None => ()
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

#[stable]
impl<A, B, I, F> DoubleEndedIterator for FilterMap<A, B, I, F> where
    I: DoubleEndedIterator<Item=A>,
    F: FnMut(A) -> Option<B>,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        for x in self.iter.by_ref().rev() {
            match (self.f)(x) {
                Some(y) => return Some(y),
                None => ()
            }
        }
        None
    }
}

/// An iterator that yields the current count and the element during iteration
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Enumerate<I> {
    iter: I,
    count: uint
}

#[stable]
impl<I> Iterator for Enumerate<I> where I: Iterator {
    type Item = (uint, <I as Iterator>::Item);

    #[inline]
    fn next(&mut self) -> Option<(uint, <I as Iterator>::Item)> {
        match self.iter.next() {
            Some(a) => {
                let ret = Some((self.count, a));
                self.count += 1;
                ret
            }
            _ => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

#[stable]
impl<I> DoubleEndedIterator for Enumerate<I> where
    I: ExactSizeIterator + DoubleEndedIterator
{
    #[inline]
    fn next_back(&mut self) -> Option<(uint, <I as Iterator>::Item)> {
        match self.iter.next_back() {
            Some(a) => {
                let len = self.iter.len();
                Some((self.count + len, a))
            }
            _ => None
        }
    }
}

#[unstable = "trait is experimental"]
impl<I> RandomAccessIterator for Enumerate<I> where I: RandomAccessIterator {
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<(uint, <I as Iterator>::Item)> {
        match self.iter.idx(index) {
            Some(a) => Some((self.count + index, a)),
            _ => None,
        }
    }
}

/// An iterator with a `peek()` that returns an optional reference to the next element.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
#[derive(Copy)]
pub struct Peekable<T, I> where I: Iterator<Item=T> {
    iter: I,
    peeked: Option<T>,
}

#[stable]
impl<T, I> Iterator for Peekable<T, I> where I: Iterator<Item=T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.peeked.is_some() { self.peeked.take() }
        else { self.iter.next() }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lo, hi) = self.iter.size_hint();
        if self.peeked.is_some() {
            let lo = lo.saturating_add(1);
            let hi = match hi {
                Some(x) => x.checked_add(1),
                None => None
            };
            (lo, hi)
        } else {
            (lo, hi)
        }
    }
}

#[stable]
impl<T, I> Peekable<T, I> where I: Iterator<Item=T> {
    /// Return a reference to the next element of the iterator with out advancing it,
    /// or None if the iterator is exhausted.
    #[inline]
    pub fn peek(&mut self) -> Option<&T> {
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
#[stable]
pub struct SkipWhile<A, I, P> where I: Iterator<Item=A>, P: FnMut(&A) -> bool {
    iter: I,
    flag: bool,
    predicate: P,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, I, P> Clone for SkipWhile<A, I, P> where
    I: Clone + Iterator<Item=A>,
    P: Clone + FnMut(&A) -> bool,
{
    fn clone(&self) -> SkipWhile<A, I, P> {
        SkipWhile {
            iter: self.iter.clone(),
            flag: self.flag,
            predicate: self.predicate.clone(),
        }
    }
}

#[stable]
impl<A, I, P> Iterator for SkipWhile<A, I, P> where I: Iterator<Item=A>, P: FnMut(&A) -> bool {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        for x in self.iter {
            if self.flag || !(self.predicate)(&x) {
                self.flag = true;
                return Some(x);
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

/// An iterator that only accepts elements while `predicate` is true
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct TakeWhile<A, I, P> where I: Iterator<Item=A>, P: FnMut(&A) -> bool {
    iter: I,
    flag: bool,
    predicate: P,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, I, P> Clone for TakeWhile<A, I, P> where
    I: Clone + Iterator<Item=A>,
    P: Clone + FnMut(&A) -> bool,
{
    fn clone(&self) -> TakeWhile<A, I, P> {
        TakeWhile {
            iter: self.iter.clone(),
            flag: self.flag,
            predicate: self.predicate.clone(),
        }
    }
}

#[stable]
impl<A, I, P> Iterator for TakeWhile<A, I, P> where I: Iterator<Item=A>, P: FnMut(&A) -> bool {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.flag {
            None
        } else {
            match self.iter.next() {
                Some(x) => {
                    if (self.predicate)(&x) {
                        Some(x)
                    } else {
                        self.flag = true;
                        None
                    }
                }
                None => None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

/// An iterator that skips over `n` elements of `iter`.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Skip<I> {
    iter: I,
    n: uint
}

#[stable]
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
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = lower.saturating_sub(self.n);

        let upper = match upper {
            Some(x) => Some(x.saturating_sub(self.n)),
            None => None
        };

        (lower, upper)
    }
}

#[unstable = "trait is experimental"]
impl<I> RandomAccessIterator for Skip<I> where I: RandomAccessIterator{
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable().saturating_sub(self.n)
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<<I as Iterator>::Item> {
        if index >= self.indexable() {
            None
        } else {
            self.iter.idx(index + self.n)
        }
    }
}

/// An iterator that only iterates over the first `n` iterations of `iter`.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Take<I> {
    iter: I,
    n: uint
}

#[stable]
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
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = cmp::min(lower, self.n);

        let upper = match upper {
            Some(x) if x < self.n => Some(x),
            _ => Some(self.n)
        };

        (lower, upper)
    }
}

#[unstable = "trait is experimental"]
impl<I> RandomAccessIterator for Take<I> where I: RandomAccessIterator{
    #[inline]
    fn indexable(&self) -> uint {
        cmp::min(self.iter.indexable(), self.n)
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<<I as Iterator>::Item> {
        if index >= self.n {
            None
        } else {
            self.iter.idx(index)
        }
    }
}


/// An iterator to maintain state while iterating another iterator
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Scan<A, B, I, St, F> where I: Iterator, F: FnMut(&mut St, A) -> Option<B> {
    iter: I,
    f: F,

    /// The current internal state to be passed to the closure next.
    pub state: St,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, B, I, St, F> Clone for Scan<A, B, I, St, F> where
    I: Clone + Iterator<Item=A>,
    St: Clone,
    F: Clone + FnMut(&mut St, A) -> Option<B>,
{
    fn clone(&self) -> Scan<A, B, I, St, F> {
        Scan {
            iter: self.iter.clone(),
            f: self.f.clone(),
            state: self.state.clone(),
        }
    }
}

#[stable]
impl<A, B, I, St, F> Iterator for Scan<A, B, I, St, F> where
    I: Iterator<Item=A>,
    F: FnMut(&mut St, A) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().and_then(|a| (self.f)(&mut self.state, a))
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the scan function
    }
}

/// An iterator that maps each element to an iterator,
/// and yields the elements of the produced iterators
///
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct FlatMap<A, B, I, U, F> where
    I: Iterator<Item=A>,
    U: Iterator<Item=B>,
    F: FnMut(A) -> U,
{
    iter: I,
    f: F,
    frontiter: Option<U>,
    backiter: Option<U>,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, B, I, U, F> Clone for FlatMap<A, B, I, U, F> where
    I: Clone + Iterator<Item=A>,
    U: Clone + Iterator<Item=B>,
    F: Clone + FnMut(A) -> U,
{
    fn clone(&self) -> FlatMap<A, B, I, U, F> {
        FlatMap {
            iter: self.iter.clone(),
            f: self.f.clone(),
            frontiter: self.frontiter.clone(),
            backiter: self.backiter.clone(),
        }
    }
}

#[stable]
impl<A, B, I, U, F> Iterator for FlatMap<A, B, I, U, F> where
    I: Iterator<Item=A>,
    U: Iterator<Item=B>,
    F: FnMut(A) -> U,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        loop {
            for inner in self.frontiter.iter_mut() {
                for x in *inner {
                    return Some(x)
                }
            }
            match self.iter.next().map(|x| (self.f)(x)) {
                None => return self.backiter.as_mut().and_then(|it| it.next()),
                next => self.frontiter = next,
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (flo, fhi) = self.frontiter.as_ref().map_or((0, Some(0)), |it| it.size_hint());
        let (blo, bhi) = self.backiter.as_ref().map_or((0, Some(0)), |it| it.size_hint());
        let lo = flo.saturating_add(blo);
        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(b)),
            _ => (lo, None)
        }
    }
}

#[stable]
impl<A, B, I, U, F> DoubleEndedIterator for FlatMap<A, B, I, U, F> where
    I: DoubleEndedIterator<Item=A>,
    U: DoubleEndedIterator<Item=B>,
    F: FnMut(A) -> U,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        loop {
            for inner in self.backiter.iter_mut() {
                match inner.next_back() {
                    None => (),
                    y => return y
                }
            }
            match self.iter.next_back().map(|x| (self.f)(x)) {
                None => return self.frontiter.as_mut().and_then(|it| it.next_back()),
                next => self.backiter = next,
            }
        }
    }
}

/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
#[derive(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Fuse<I> {
    iter: I,
    done: bool
}

#[stable]
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
    fn size_hint(&self) -> (uint, Option<uint>) {
        if self.done {
            (0, Some(0))
        } else {
            self.iter.size_hint()
        }
    }
}

#[stable]
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
#[unstable = "trait is experimental"]
impl<I> RandomAccessIterator for Fuse<I> where I: RandomAccessIterator {
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<<I as Iterator>::Item> {
        self.iter.idx(index)
    }
}

impl<I> Fuse<I> {
    /// Resets the fuse such that the next call to .next() or .next_back() will
    /// call the underlying iterator again even if it previously returned None.
    #[inline]
    #[unstable = "seems marginal"]
    pub fn reset_fuse(&mut self) {
        self.done = false
    }
}

/// An iterator that calls a function with a reference to each
/// element before yielding it.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[stable]
pub struct Inspect<A, I, F> where I: Iterator<Item=A>, F: FnMut(&A) {
    iter: I,
    f: F,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, I, F> Clone for Inspect<A, I, F> where
    I: Clone + Iterator<Item=A>,
    F: Clone + FnMut(&A),
{
    fn clone(&self) -> Inspect<A, I, F> {
        Inspect {
            iter: self.iter.clone(),
            f: self.f.clone(),
        }
    }
}

impl<A, I, F> Inspect<A, I, F> where I: Iterator<Item=A>, F: FnMut(&A) {
    #[inline]
    fn do_inspect(&mut self, elt: Option<A>) -> Option<A> {
        match elt {
            Some(ref a) => (self.f)(a),
            None => ()
        }

        elt
    }
}

#[stable]
impl<A, I, F> Iterator for Inspect<A, I, F> where I: Iterator<Item=A>, F: FnMut(&A) {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let next = self.iter.next();
        self.do_inspect(next)
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

#[stable]
impl<A, I, F> DoubleEndedIterator for Inspect<A, I, F> where
    I: DoubleEndedIterator<Item=A>,
    F: FnMut(&A),
{
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        let next = self.iter.next_back();
        self.do_inspect(next)
    }
}

#[unstable = "trait is experimental"]
impl<A, I, F> RandomAccessIterator for Inspect<A, I, F> where
    I: RandomAccessIterator<Item=A>,
    F: FnMut(&A),
{
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
        let element = self.iter.idx(index);
        self.do_inspect(element)
    }
}

/// An iterator that passes mutable state to a closure and yields the result.
///
/// # Example: The Fibonacci Sequence
///
/// An iterator that yields sequential Fibonacci numbers, and stops on overflow.
///
/// ```rust
/// use std::iter::Unfold;
/// use std::num::Int; // For `.checked_add()`
///
/// // This iterator will yield up to the last Fibonacci number before the max value of `u32`.
/// // You can simply change `u32` to `u64` in this line if you want higher values than that.
/// let mut fibonacci = Unfold::new((Some(0u32), Some(1u32)), |&mut (ref mut x2, ref mut x1)| {
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
#[unstable]
pub struct Unfold<A, St, F> where F: FnMut(&mut St) -> Option<A> {
    f: F,
    /// Internal state that will be passed to the closure on the next iteration
    pub state: St,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable]
impl<A, St, F> Clone for Unfold<A, St, F> where
    F: Clone + FnMut(&mut St) -> Option<A>,
    St: Clone,
{
    fn clone(&self) -> Unfold<A, St, F> {
        Unfold {
            f: self.f.clone(),
            state: self.state.clone(),
        }
    }
}

#[unstable]
impl<A, St, F> Unfold<A, St, F> where F: FnMut(&mut St) -> Option<A> {
    /// Creates a new iterator with the specified closure as the "iterator
    /// function" and an initial state to eventually pass to the closure
    #[inline]
    pub fn new(initial_state: St, f: F) -> Unfold<A, St, F> {
        Unfold {
            f: f,
            state: initial_state
        }
    }
}

#[stable]
impl<A, St, F> Iterator for Unfold<A, St, F> where F: FnMut(&mut St) -> Option<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        (self.f)(&mut self.state)
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        // no possible known bounds at this point
        (0, None)
    }
}

/// An infinite iterator starting at `start` and advancing by `step` with each
/// iteration
#[derive(Clone, Copy)]
#[unstable = "may be renamed or replaced by range notation adapaters"]
pub struct Counter<A> {
    /// The current state the counter is at (next value to be yielded)
    state: A,
    /// The amount that this iterator is stepping by
    step: A,
}

/// Creates a new counter with the specified start/step
#[inline]
#[unstable = "may be renamed or replaced by range notation adapaters"]
pub fn count<A>(start: A, step: A) -> Counter<A> {
    Counter{state: start, step: step}
}

#[stable]
impl<A: Add<Output=A> + Clone> Iterator for Counter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let result = self.state.clone();
        self.state = self.state.clone() + self.step.clone();
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (uint::MAX, None) // Too bad we can't specify an infinite lower bound
    }
}

/// An iterator over the range [start, stop)
#[derive(Clone, Copy)]
#[unstable = "will be replaced by range notation"]
pub struct Range<A> {
    state: A,
    stop: A,
    one: A,
}

/// Returns an iterator over the given range [start, stop) (that is, starting
/// at start (inclusive), and ending at stop (exclusive)).
///
/// # Example
///
/// ```rust
/// let array = [0, 1, 2, 3, 4];
///
/// for i in range(0, 5u) {
///     println!("{}", i);
///     assert_eq!(i,  array[i]);
/// }
/// ```
#[inline]
#[unstable = "will be replaced by range notation"]
pub fn range<A: Int>(start: A, stop: A) -> Range<A> {
    Range {
        state: start,
        stop: stop,
        one: Int::one(),
    }
}

// FIXME: #10414: Unfortunate type bound
#[unstable = "will be replaced by range notation"]
impl<A: Int + ToPrimitive> Iterator for Range<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.state < self.stop {
            let result = self.state.clone();
            self.state = self.state + self.one;
            Some(result)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        // This first checks if the elements are representable as i64. If they aren't, try u64 (to
        // handle cases like range(huge, huger)). We don't use uint/int because the difference of
        // the i64/u64 might lie within their range.
        let bound = match self.state.to_i64() {
            Some(a) => {
                let sz = self.stop.to_i64().map(|b| b.checked_sub(a));
                match sz {
                    Some(Some(bound)) => bound.to_uint(),
                    _ => None,
                }
            },
            None => match self.state.to_u64() {
                Some(a) => {
                    let sz = self.stop.to_u64().map(|b| b.checked_sub(a));
                    match sz {
                        Some(Some(bound)) => bound.to_uint(),
                        _ => None
                    }
                },
                None => None
            }
        };

        match bound {
            Some(b) => (b, Some(b)),
            // Standard fallback for unbounded/unrepresentable bounds
            None => (0, None)
        }
    }
}

/// `Int` is required to ensure the range will be the same regardless of
/// the direction it is consumed.
#[unstable = "will be replaced by range notation"]
impl<A: Int + ToPrimitive> DoubleEndedIterator for Range<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.stop > self.state {
            self.stop = self.stop - self.one;
            Some(self.stop.clone())
        } else {
            None
        }
    }
}

/// An iterator over the range [start, stop]
#[derive(Clone)]
#[unstable = "likely to be replaced by range notation and adapters"]
pub struct RangeInclusive<A> {
    range: Range<A>,
    done: bool,
}

/// Return an iterator over the range [start, stop]
#[inline]
#[unstable = "likely to be replaced by range notation and adapters"]
pub fn range_inclusive<A: Int>(start: A, stop: A) -> RangeInclusive<A> {
    RangeInclusive {
        range: range(start, stop),
        done: false,
    }
}

#[unstable = "likely to be replaced by range notation and adapters"]
impl<A: Int + ToPrimitive> Iterator for RangeInclusive<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        match self.range.next() {
            Some(x) => Some(x),
            None => {
                if !self.done && self.range.state == self.range.stop {
                    self.done = true;
                    Some(self.range.stop.clone())
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lo, hi) = self.range.size_hint();
        if self.done {
            (lo, hi)
        } else {
            let lo = lo.saturating_add(1);
            let hi = match hi {
                Some(x) => x.checked_add(1),
                None => None
            };
            (lo, hi)
        }
    }
}

#[unstable = "likely to be replaced by range notation and adapters"]
impl<A: Int + ToPrimitive> DoubleEndedIterator for RangeInclusive<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.range.stop > self.range.state {
            let result = self.range.stop.clone();
            self.range.stop = self.range.stop - self.range.one;
            Some(result)
        } else if !self.done && self.range.state == self.range.stop {
            self.done = true;
            Some(self.range.stop.clone())
        } else {
            None
        }
    }
}

/// An iterator over the range [start, stop) by `step`. It handles overflow by stopping.
#[derive(Clone)]
#[unstable = "likely to be replaced by range notation and adapters"]
pub struct RangeStep<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
}

/// Return an iterator over the range [start, stop) by `step`. It handles overflow by stopping.
#[inline]
#[unstable = "likely to be replaced by range notation and adapters"]
pub fn range_step<A: Int>(start: A, stop: A, step: A) -> RangeStep<A> {
    let rev = step < Int::zero();
    RangeStep{state: start, stop: stop, step: step, rev: rev}
}

#[unstable = "likely to be replaced by range notation and adapters"]
impl<A: Int> Iterator for RangeStep<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if (self.rev && self.state > self.stop) || (!self.rev && self.state < self.stop) {
            let result = self.state;
            match self.state.checked_add(self.step) {
                Some(x) => self.state = x,
                None => self.state = self.stop.clone()
            }
            Some(result)
        } else {
            None
        }
    }
}

/// An iterator over the range [start, stop] by `step`. It handles overflow by stopping.
#[derive(Clone)]
#[unstable = "likely to be replaced by range notation and adapters"]
pub struct RangeStepInclusive<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
    done: bool,
}

/// Return an iterator over the range [start, stop] by `step`. It handles overflow by stopping.
#[inline]
#[unstable = "likely to be replaced by range notation and adapters"]
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

#[unstable = "likely to be replaced by range notation and adapters"]
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


/// The `Step` trait identifies objects which can be stepped over in both
/// directions. The `steps_between` function provides a way to
/// compare two Step objects (it could be provided using `step()` and `Ord`,
/// but the implementation would be so inefficient as to be useless).
#[unstable = "design of range notation/iteration is in flux"]
pub trait Step: Ord {
    /// Change self to the next object.
    fn step(&mut self);
    /// Change self to the previous object.
    fn step_back(&mut self);
    /// The steps_between two step objects.
    /// start should always be less than end, so the result should never be negative.
    /// Return None if it is not possible to calculate steps_between without
    /// overflow.
    fn steps_between(start: &Self, end: &Self) -> Option<uint>;
}

macro_rules! step_impl {
    ($($t:ty)*) => ($(
        #[unstable = "Trait is unstable."]
        impl Step for $t {
            #[inline]
            fn step(&mut self) { *self += 1; }
            #[inline]
            fn step_back(&mut self) { *self -= 1; }
            #[inline]
            fn steps_between(start: &$t, end: &$t) -> Option<uint> {
                debug_assert!(end >= start);
                Some((*end - *start) as uint)
            }
        }
    )*)
}

macro_rules! step_impl_no_between {
    ($($t:ty)*) => ($(
        #[unstable = "Trait is unstable."]
        impl Step for $t {
            #[inline]
            fn step(&mut self) { *self += 1; }
            #[inline]
            fn step_back(&mut self) { *self -= 1; }
            #[inline]
            fn steps_between(_start: &$t, _end: &$t) -> Option<uint> {
                None
            }
        }
    )*)
}

step_impl!(uint u8 u16 u32 int i8 i16 i32);
#[cfg(any(all(stage0, target_word_size = "64"), all(not(stage0), target_pointer_width = "64")))]
step_impl!(u64 i64);
#[cfg(any(all(stage0, target_word_size = "32"), all(not(stage0), target_pointer_width = "32")))]
step_impl_no_between!(u64 i64);


/// An iterator that repeats an element endlessly
#[derive(Clone)]
#[stable]
pub struct Repeat<A> {
    element: A
}

#[stable]
impl<A: Clone> Iterator for Repeat<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> { self.idx(0) }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { (uint::MAX, None) }
}

#[stable]
impl<A: Clone> DoubleEndedIterator for Repeat<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.idx(0) }
}

#[unstable = "trait is experimental"]
impl<A: Clone> RandomAccessIterator for Repeat<A> {
    #[inline]
    fn indexable(&self) -> uint { uint::MAX }
    #[inline]
    fn idx(&mut self, _: uint) -> Option<A> { Some(self.element.clone()) }
}

type IterateState<T, F> = (F, Option<T>, bool);

/// An iterator that repeatedly applies a given function, starting
/// from a given seed value.
#[unstable]
pub type Iterate<T, F> = Unfold<T, IterateState<T, F>, fn(&mut IterateState<T, F>) -> Option<T>>;

/// Create a new iterator that produces an infinite sequence of
/// repeated applications of the given function `f`.
#[unstable]
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
        } else {
            match val.take() {
                Some(x) => {
                    *val = Some((*f)(x))
                }
                None => {}
            }
        }
        val.clone()
    }

    // coerce to a fn pointer
    let next: fn(&mut IterateState<T,F>) -> Option<T> = next;

    Unfold::new((f, Some(seed), true), next)
}

/// Create a new iterator that endlessly repeats the element `elt`.
#[inline]
#[stable]
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
#[unstable = "needs review and revision"]
pub mod order {
    use cmp;
    use cmp::{Eq, Ord, PartialOrd, PartialEq};
    use cmp::Ordering::{Equal, Less, Greater};
    use option::Option;
    use option::Option::{Some, None};
    use super::Iterator;

    /// Compare `a` and `b` for equality using `Eq`
    pub fn equals<A, T, S>(mut a: T, mut b: S) -> bool where
        A: Eq,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
    pub fn cmp<A, T, S>(mut a: T, mut b: S) -> cmp::Ordering where
        A: Ord,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
    pub fn partial_cmp<A, T, S>(mut a: T, mut b: S) -> Option<cmp::Ordering> where
        A: PartialOrd,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
    pub fn eq<A, B, L, R>(mut a: L, mut b: R) -> bool where
        A: PartialEq<B>,
        L: Iterator<Item=A>,
        R: Iterator<Item=B>,
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
    pub fn ne<A, B, L, R>(mut a: L, mut b: R) -> bool where
        A: PartialEq<B>,
        L: Iterator<Item=A>,
        R: Iterator<Item=B>,
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
    pub fn lt<A, T, S>(mut a: T, mut b: S) -> bool where
        A: PartialOrd,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
    pub fn le<A, T, S>(mut a: T, mut b: S) -> bool where
        A: PartialOrd,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
    pub fn gt<A, T, S>(mut a: T, mut b: S) -> bool where
        A: PartialOrd,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
    pub fn ge<A, T, S>(mut a: T, mut b: S) -> bool where
        A: PartialOrd,
        T: Iterator<Item=A>,
        S: Iterator<Item=A>,
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
