// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Composable external iterators

# The `Iterator` trait

This module defines Rust's core iteration trait. The `Iterator` trait has one
unimplemented method, `next`. All other methods are derived through default
methods to perform operations such as `zip`, `chain`, `enumerate`, and `fold`.

The goal of this module is to unify iteration across all containers in Rust.
An iterator can be considered as a state machine which is used to track which
element will be yielded next.

There are various extensions also defined in this module to assist with various
types of iteration, such as the `DoubleEndedIterator` for iterating in reverse,
the `FromIterator` trait for creating a container from an iterator, and much
more.

## Rust's `for` loop

The special syntax used by rust's `for` loop is based around the `Iterator`
trait defined in this module. For loops can be viewed as a syntactical expansion
into a `loop`, for example, the `for` loop in this example is essentially
translated to the `loop` below.

```rust
let values = vec![1i, 2, 3];

// "Syntactical sugar" taking advantage of an iterator
for &x in values.iter() {
    println!("{}", x);
}

// Rough translation of the iteration without a `for` iterator.
let mut it = values.iter();
loop {
    match it.next() {
        Some(&x) => {
            println!("{}", x);
        }
        None => { break }
    }
}
```

This `for` loop syntax can be applied to any iterator over any type.

## Iteration protocol and more

More detailed information about iterators can be found in the [container
guide](http://doc.rust-lang.org/guide-container.html) with
the rest of the rust manuals.

*/

use clone::Clone;
use cmp;
use cmp::{PartialEq, PartialOrd, Ord};
use mem;
use num::{Zero, One, CheckedAdd, CheckedSub, Saturating, ToPrimitive, Int};
use ops::{Add, Mul, Sub};
use option::{Option, Some, None};
use uint;

/// Conversion from an `Iterator`
pub trait FromIterator<A> {
    /// Build a container with elements from an external iterator.
    fn from_iter<T: Iterator<A>>(iterator: T) -> Self;
}

/// A type growable from an `Iterator` implementation
pub trait Extendable<A>: FromIterator<A> {
    /// Extend a container with the elements yielded by an iterator
    fn extend<T: Iterator<A>>(&mut self, iterator: T);
}

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
pub trait Iterator<A> {
    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    fn next(&mut self) -> Option<A>;

    /// Returns a lower and upper bound on the remaining length of the iterator.
    ///
    /// An upper bound of `None` means either there is no known upper bound, or the upper bound
    /// does not fit within a `uint`.
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { (0, None) }

    /// Chain this iterator with another, returning a new iterator which will
    /// finish iterating over the current iterator, and then it will iterate
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
    fn chain<U: Iterator<A>>(self, other: U) -> Chain<Self, U> {
        Chain{a: self, b: other, flag: false}
    }

    /// Creates an iterator which iterates over both this and the specified
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
    fn zip<B, U: Iterator<B>>(self, other: U) -> Zip<Self, U> {
        Zip{a: self, b: other}
    }

    /// Creates a new iterator which will apply the specified function to each
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
    fn map<'r, B>(self, f: |A|: 'r -> B) -> Map<'r, A, B, Self> {
        Map{iter: self, f: f}
    }

    /// Creates an iterator which applies the predicate to each element returned
    /// by this iterator. Only elements which have the predicate evaluate to
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
    fn filter<'r>(self, predicate: |&A|: 'r -> bool) -> Filter<'r, A, Self> {
        Filter{iter: self, predicate: predicate}
    }

    /// Creates an iterator which both filters and maps elements.
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
    fn filter_map<'r, B>(self, f: |A|: 'r -> Option<B>) -> FilterMap<'r, A, B, Self> {
        FilterMap { iter: self, f: f }
    }

    /// Creates an iterator which yields a pair of the value returned by this
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
    fn peekable(self) -> Peekable<A, Self> {
        Peekable{iter: self, peeked: None}
    }

    /// Creates an iterator which invokes the predicate on elements until it
    /// returns false. Once the predicate returns false, all further elements are
    /// yielded.
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
    fn skip_while<'r>(self, predicate: |&A|: 'r -> bool) -> SkipWhile<'r, A, Self> {
        SkipWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator which yields elements so long as the predicate
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
    fn take_while<'r>(self, predicate: |&A|: 'r -> bool) -> TakeWhile<'r, A, Self> {
        TakeWhile{iter: self, flag: false, predicate: predicate}
    }

    /// Creates an iterator which skips the first `n` elements of this iterator,
    /// and then it yields all further items.
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
    fn skip(self, n: uint) -> Skip<Self> {
        Skip{iter: self, n: n}
    }

    /// Creates an iterator which yields the first `n` elements of this
    /// iterator, and then it will always return None.
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
    fn take(self, n: uint) -> Take<Self> {
        Take{iter: self, n: n}
    }

    /// Creates a new iterator which behaves in a similar fashion to fold.
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
    fn scan<'r, St, B>(self, initial_state: St, f: |&mut St, A|: 'r -> Option<B>)
        -> Scan<'r, A, B, Self, St> {
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
    fn flat_map<'r, B, U: Iterator<B>>(self, f: |A|: 'r -> U)
        -> FlatMap<'r, A, Self, U> {
        FlatMap{iter: self, f: f, frontiter: None, backiter: None }
    }

    /// Creates an iterator that yields `None` forever after the underlying
    /// iterator yields `None`. Random-access iterator behavior is not
    /// affected, only single and double-ended iterator behavior.
    ///
    /// # Example
    ///
    /// ```rust
    /// fn process<U: Iterator<int>>(it: U) -> int {
    ///     let mut it = it.fuse();
    ///     let mut sum = 0;
    ///     for x in it {
    ///         if x > 5 {
    ///             continue;
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
    /// assert_eq!(process(x.move_iter()), 1006);
    /// ```
    #[inline]
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
    fn inspect<'r>(self, f: |&A|: 'r) -> Inspect<'r, A, Self> {
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
    fn by_ref<'r>(&'r mut self) -> ByRef<'r, Self> {
        ByRef{iter: self}
    }

    /// Apply a function to each element, or stop iterating if the
    /// function returns `false`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// range(0u, 5).advance(|x| {print!("{} ", x); true});
    /// ```
    #[deprecated = "use the `all` method instead"]
    #[inline]
    fn advance(&mut self, f: |A| -> bool) -> bool {
        loop {
            match self.next() {
                Some(x) => {
                    if !f(x) { return false; }
                }
                None => { return true; }
            }
        }
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
    fn collect<B: FromIterator<A>>(&mut self) -> B {
        FromIterator::from_iter(self.by_ref())
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
    fn nth(&mut self, mut n: uint) -> Option<A> {
        for x in *self {
            if n == 0 { return Some(x) }
            n -= 1;
        }
        None
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
    fn last(&mut self) -> Option<A> {
        let mut last = None;
        for x in *self { last = Some(x); }
        last
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
    fn fold<B>(&mut self, init: B, f: |B, A| -> B) -> B {
        let mut accum = init;
        for x in *self {
            accum = f(accum, x);
        }
        accum
    }

    /// Counts the number of elements in this iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.count() == 5);
    /// assert!(it.count() == 0);
    /// ```
    #[inline]
    fn count(&mut self) -> uint {
        self.fold(0, |cnt, _x| cnt + 1)
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
    fn all(&mut self, f: |A| -> bool) -> bool {
        for x in *self { if !f(x) { return false; } }
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
    fn any(&mut self, f: |A| -> bool) -> bool {
        for x in *self { if f(x) { return true; } }
        false
    }

    /// Return the first element satisfying the specified predicate
    #[inline]
    fn find(&mut self, predicate: |&A| -> bool) -> Option<A> {
        for x in *self {
            if predicate(&x) { return Some(x) }
        }
        None
    }

    /// Return the index of the first element satisfying the specified predicate
    #[inline]
    fn position(&mut self, predicate: |A| -> bool) -> Option<uint> {
        let mut i = 0;
        for x in *self {
            if predicate(x) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Return the element that gives the maximum value from the
    /// specified function.
    ///
    /// # Example
    ///
    /// ```rust
    /// let xs = [-3i, 0, 1, 5, -10];
    /// assert_eq!(*xs.iter().max_by(|x| x.abs()).unwrap(), -10);
    /// ```
    #[inline]
    fn max_by<B: Ord>(&mut self, f: |&A| -> B) -> Option<A> {
        self.fold(None, |max: Option<(A, B)>, x| {
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
    /// let xs = [-3i, 0, 1, 5, -10];
    /// assert_eq!(*xs.iter().min_by(|x| x.abs()).unwrap(), 0);
    /// ```
    #[inline]
    fn min_by<B: Ord>(&mut self, f: |&A| -> B) -> Option<A> {
        self.fold(None, |min: Option<(A, B)>, x| {
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
}

/// A range iterator able to yield elements from both ends
///
/// A `DoubleEndedIterator` can be thought of as a deque in that `next()` and `next_back()` exhaust
/// elements from the *same* range, and do not work independently of each other.
pub trait DoubleEndedIterator<A>: Iterator<A> {
    /// Yield an element from the end of the range, returning `None` if the range is empty.
    fn next_back(&mut self) -> Option<A>;

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
    fn rev(self) -> Rev<Self> {
        Rev{iter: self}
    }
}

/// A double-ended iterator yielding mutable references
pub trait MutableDoubleEndedIterator {
    // FIXME: #5898: should be called `reverse`
    /// Use an iterator to reverse a container in-place
    fn reverse_(&mut self);
}

impl<'a, A, T: DoubleEndedIterator<&'a mut A>> MutableDoubleEndedIterator for T {
    // FIXME: #5898: should be called `reverse`
    /// Use an iterator to reverse a container in-place
    fn reverse_(&mut self) {
        loop {
            match (self.next(), self.next_back()) {
                (Some(x), Some(y)) => mem::swap(x, y),
                _ => break
            }
        }
    }
}


/// An object implementing random access indexing by `uint`
///
/// A `RandomAccessIterator` should be either infinite or a `DoubleEndedIterator`.
/// Calling `next()` or `next_back()` on a `RandomAccessIterator`
/// reduces the indexable range accordingly. That is, `it.idx(1)` will become `it.idx(0)`
/// after `it.next()` is called.
pub trait RandomAccessIterator<A>: Iterator<A> {
    /// Return the number of indexable elements. At most `std::uint::MAX`
    /// elements are indexable, even if the iterator represents a longer range.
    fn indexable(&self) -> uint;

    /// Return an element at an index, or `None` if the index is out of bounds
    fn idx(&mut self, index: uint) -> Option<A>;
}

/// An iterator that knows its exact length
///
/// This trait is a helper for iterators like the vector iterator, so that
/// it can support double-ended enumeration.
///
/// `Iterator::size_hint` *must* return the exact size of the iterator.
/// Note that the size must fit in `uint`.
pub trait ExactSize<A> : DoubleEndedIterator<A> {
    /// Return the index of the last element satisfying the specified predicate
    ///
    /// If no element matches, None is returned.
    #[inline]
    fn rposition(&mut self, predicate: |A| -> bool) -> Option<uint> {
        let len = self.len();
        for i in range(0, len).rev() {
            if predicate(self.next_back().expect("rposition: incorrect ExactSize")) {
                return Some(i);
            }
        }
        None
    }

    #[inline]
    /// Return the exact length of the iterator.
    fn len(&self) -> uint {
        let (lower, upper) = self.size_hint();
        assert_eq!(upper, Some(lower));
        lower
    }
}

// All adaptors that preserve the size of the wrapped iterator are fine
// Adaptors that may overflow in `size_hint` are not, i.e. `Chain`.
impl<A, T: ExactSize<A>> ExactSize<(uint, A)> for Enumerate<T> {}
impl<'a, A, T: ExactSize<A>> ExactSize<A> for Inspect<'a, A, T> {}
impl<A, T: ExactSize<A>> ExactSize<A> for Rev<T> {}
impl<'a, A, B, T: ExactSize<A>> ExactSize<B> for Map<'a, A, B, T> {}
impl<A, B, T: ExactSize<A>, U: ExactSize<B>> ExactSize<(A, B)> for Zip<T, U> {}

/// An double-ended iterator with the direction inverted
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Rev<T> {
    iter: T
}

impl<A, T: DoubleEndedIterator<A>> Iterator<A> for Rev<T> {
    #[inline]
    fn next(&mut self) -> Option<A> { self.iter.next_back() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

impl<A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for Rev<T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.iter.next() }
}

impl<A, T: DoubleEndedIterator<A> + RandomAccessIterator<A>> RandomAccessIterator<A>
    for Rev<T> {
    #[inline]
    fn indexable(&self) -> uint { self.iter.indexable() }
    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
        let amt = self.indexable();
        self.iter.idx(amt - index - 1)
    }
}

/// A mutable reference to an iterator
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct ByRef<'a, T> {
    iter: &'a mut T
}

impl<'a, A, T: Iterator<A>> Iterator<A> for ByRef<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<A> { self.iter.next() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

impl<'a, A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for ByRef<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.iter.next_back() }
}

/// A trait for iterators over elements which can be added together
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
    fn sum(&mut self) -> A;
}

impl<A: Add<A, A> + Zero, T: Iterator<A>> AdditiveIterator<A> for T {
    #[inline]
    fn sum(&mut self) -> A {
        let zero: A = Zero::zero();
        self.fold(zero, |s, x| s + x)
    }
}

/// A trait for iterators over elements whose elements can be multiplied
/// together.
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
    fn product(&mut self) -> A;
}

impl<A: Mul<A, A> + One, T: Iterator<A>> MultiplicativeIterator<A> for T {
    #[inline]
    fn product(&mut self) -> A {
        let one: A = One::one();
        self.fold(one, |p, x| p * x)
    }
}

/// A trait for iterators over elements which can be compared to one another.
/// The type of each element must ascribe to the `PartialOrd` trait.
pub trait OrdIterator<A> {
    /// Consumes the entire iterator to return the maximum element.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().max().unwrap() == &5);
    /// ```
    fn max(&mut self) -> Option<A>;

    /// Consumes the entire iterator to return the minimum element.
    ///
    /// # Example
    ///
    /// ```rust
    /// let a = [1i, 2, 3, 4, 5];
    /// assert!(a.iter().min().unwrap() == &1);
    /// ```
    fn min(&mut self) -> Option<A>;

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
    /// and so faster than calling `min` and `max separately which does `2 * n` comparisons.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::{NoElements, OneElement, MinMax};
    ///
    /// let v: [int, ..0] = [];
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
    fn min_max(&mut self) -> MinMaxResult<A>;
}

impl<A: Ord, T: Iterator<A>> OrdIterator<A> for T {
    #[inline]
    fn max(&mut self) -> Option<A> {
        self.fold(None, |max, x| {
            match max {
                None    => Some(x),
                Some(y) => Some(cmp::max(x, y))
            }
        })
    }

    #[inline]
    fn min(&mut self) -> Option<A> {
        self.fold(None, |min, x| {
            match min {
                None    => Some(x),
                Some(y) => Some(cmp::min(x, y))
            }
        })
    }

    fn min_max(&mut self) -> MinMaxResult<A> {
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
}

/// `MinMaxResult` is an enum returned by `min_max`. See `OrdIterator::min_max` for more detail.
#[deriving(Clone, PartialEq, Show)]
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
    /// use std::iter::{NoElements, OneElement, MinMax, MinMaxResult};
    ///
    /// let r: MinMaxResult<int> = NoElements;
    /// assert_eq!(r.into_option(), None)
    ///
    /// let r = OneElement(1i);
    /// assert_eq!(r.into_option(), Some((1,1)));
    ///
    /// let r = MinMax(1i,2i);
    /// assert_eq!(r.into_option(), Some((1,2)));
    /// ```
    pub fn into_option(self) -> Option<(T,T)> {
        match self {
            NoElements => None,
            OneElement(x) => Some((x.clone(), x)),
            MinMax(x, y) => Some((x, y))
        }
    }
}

/// A trait for iterators that are cloneable.
pub trait CloneableIterator {
    /// Repeats an iterator endlessly
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::iter::{CloneableIterator, count};
    ///
    /// let a = count(1i,1i).take(1);
    /// let mut cy = a.cycle();
    /// assert_eq!(cy.next(), Some(1));
    /// assert_eq!(cy.next(), Some(1));
    /// ```
    fn cycle(self) -> Cycle<Self>;
}

impl<A, T: Clone + Iterator<A>> CloneableIterator for T {
    #[inline]
    fn cycle(self) -> Cycle<T> {
        Cycle{orig: self.clone(), iter: self}
    }
}

/// An iterator that repeats endlessly
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Cycle<T> {
    orig: T,
    iter: T,
}

impl<A, T: Clone + Iterator<A>> Iterator<A> for Cycle<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
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

impl<A, T: Clone + RandomAccessIterator<A>> RandomAccessIterator<A> for Cycle<T> {
    #[inline]
    fn indexable(&self) -> uint {
        if self.orig.indexable() > 0 {
            uint::MAX
        } else {
            0
        }
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
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

/// An iterator which strings two iterators together
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Chain<T, U> {
    a: T,
    b: U,
    flag: bool,
}

impl<A, T: Iterator<A>, U: Iterator<A>> Iterator<A> for Chain<T, U> {
    #[inline]
    fn next(&mut self) -> Option<A> {
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
            (Some(x), Some(y)) => x.checked_add(&y),
            _ => None
        };

        (lower, upper)
    }
}

impl<A, T: DoubleEndedIterator<A>, U: DoubleEndedIterator<A>> DoubleEndedIterator<A>
for Chain<T, U> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        match self.b.next_back() {
            Some(x) => Some(x),
            None => self.a.next_back()
        }
    }
}

impl<A, T: RandomAccessIterator<A>, U: RandomAccessIterator<A>> RandomAccessIterator<A>
for Chain<T, U> {
    #[inline]
    fn indexable(&self) -> uint {
        let (a, b) = (self.a.indexable(), self.b.indexable());
        a.saturating_add(b)
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
        let len = self.a.indexable();
        if index < len {
            self.a.idx(index)
        } else {
            self.b.idx(index - len)
        }
    }
}

/// An iterator which iterates two other iterators simultaneously
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Zip<T, U> {
    a: T,
    b: U
}

impl<A, B, T: Iterator<A>, U: Iterator<B>> Iterator<(A, B)> for Zip<T, U> {
    #[inline]
    fn next(&mut self) -> Option<(A, B)> {
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

impl<A, B, T: ExactSize<A>, U: ExactSize<B>> DoubleEndedIterator<(A, B)>
for Zip<T, U> {
    #[inline]
    fn next_back(&mut self) -> Option<(A, B)> {
        let (a_sz, a_upper) = self.a.size_hint();
        let (b_sz, b_upper) = self.b.size_hint();
        assert!(a_upper == Some(a_sz));
        assert!(b_upper == Some(b_sz));
        if a_sz < b_sz {
            for _ in range(0, b_sz - a_sz) { self.b.next_back(); }
        } else if a_sz > b_sz {
            for _ in range(0, a_sz - b_sz) { self.a.next_back(); }
        }
        let (a_sz, _) = self.a.size_hint();
        let (b_sz, _) = self.b.size_hint();
        assert!(a_sz == b_sz);
        match (self.a.next_back(), self.b.next_back()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None
        }
    }
}

impl<A, B, T: RandomAccessIterator<A>, U: RandomAccessIterator<B>>
RandomAccessIterator<(A, B)> for Zip<T, U> {
    #[inline]
    fn indexable(&self) -> uint {
        cmp::min(self.a.indexable(), self.b.indexable())
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<(A, B)> {
        match self.a.idx(index) {
            None => None,
            Some(x) => match self.b.idx(index) {
                None => None,
                Some(y) => Some((x, y))
            }
        }
    }
}

/// An iterator which maps the values of `iter` with `f`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Map<'a, A, B, T> {
    iter: T,
    f: |A|: 'a -> B
}

impl<'a, A, B, T> Map<'a, A, B, T> {
    #[inline]
    fn do_map(&mut self, elt: Option<A>) -> Option<B> {
        match elt {
            Some(a) => Some((self.f)(a)),
            _ => None
        }
    }
}

impl<'a, A, B, T: Iterator<A>> Iterator<B> for Map<'a, A, B, T> {
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

impl<'a, A, B, T: DoubleEndedIterator<A>> DoubleEndedIterator<B> for Map<'a, A, B, T> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        let next = self.iter.next_back();
        self.do_map(next)
    }
}

impl<'a, A, B, T: RandomAccessIterator<A>> RandomAccessIterator<B> for Map<'a, A, B, T> {
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

/// An iterator which filters the elements of `iter` with `predicate`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Filter<'a, A, T> {
    iter: T,
    predicate: |&A|: 'a -> bool
}

impl<'a, A, T: Iterator<A>> Iterator<A> for Filter<'a, A, T> {
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

impl<'a, A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for Filter<'a, A, T> {
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

/// An iterator which uses `f` to both filter and map elements from `iter`
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct FilterMap<'a, A, B, T> {
    iter: T,
    f: |A|: 'a -> Option<B>
}

impl<'a, A, B, T: Iterator<A>> Iterator<B> for FilterMap<'a, A, B, T> {
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

impl<'a, A, B, T: DoubleEndedIterator<A>> DoubleEndedIterator<B>
for FilterMap<'a, A, B, T> {
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

/// An iterator which yields the current count and the element during iteration
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Enumerate<T> {
    iter: T,
    count: uint
}

impl<A, T: Iterator<A>> Iterator<(uint, A)> for Enumerate<T> {
    #[inline]
    fn next(&mut self) -> Option<(uint, A)> {
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

impl<A, T: ExactSize<A>> DoubleEndedIterator<(uint, A)> for Enumerate<T> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, A)> {
        match self.iter.next_back() {
            Some(a) => {
                let (lower, upper) = self.iter.size_hint();
                assert!(upper == Some(lower));
                Some((self.count + lower, a))
            }
            _ => None
        }
    }
}

impl<A, T: RandomAccessIterator<A>> RandomAccessIterator<(uint, A)> for Enumerate<T> {
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<(uint, A)> {
        match self.iter.idx(index) {
            Some(a) => Some((self.count + index, a)),
            _ => None,
        }
    }
}

/// An iterator with a `peek()` that returns an optional reference to the next element.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Peekable<A, T> {
    iter: T,
    peeked: Option<A>,
}

impl<A, T: Iterator<A>> Iterator<A> for Peekable<A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.peeked.is_some() { self.peeked.take() }
        else { self.iter.next() }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lo, hi) = self.iter.size_hint();
        if self.peeked.is_some() {
            let lo = lo.saturating_add(1);
            let hi = match hi {
                Some(x) => x.checked_add(&1),
                None => None
            };
            (lo, hi)
        } else {
            (lo, hi)
        }
    }
}

impl<'a, A, T: Iterator<A>> Peekable<A, T> {
    /// Return a reference to the next element of the iterator with out advancing it,
    /// or None if the iterator is exhausted.
    #[inline]
    pub fn peek(&'a mut self) -> Option<&'a A> {
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

/// An iterator which rejects elements while `predicate` is true
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct SkipWhile<'a, A, T> {
    iter: T,
    flag: bool,
    predicate: |&A|: 'a -> bool
}

impl<'a, A, T: Iterator<A>> Iterator<A> for SkipWhile<'a, A, T> {
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

/// An iterator which only accepts elements while `predicate` is true
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct TakeWhile<'a, A, T> {
    iter: T,
    flag: bool,
    predicate: |&A|: 'a -> bool
}

impl<'a, A, T: Iterator<A>> Iterator<A> for TakeWhile<'a, A, T> {
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

/// An iterator which skips over `n` elements of `iter`.
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Skip<T> {
    iter: T,
    n: uint
}

impl<A, T: Iterator<A>> Iterator<A> for Skip<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
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

impl<A, T: RandomAccessIterator<A>> RandomAccessIterator<A> for Skip<T> {
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable().saturating_sub(self.n)
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
        if index >= self.indexable() {
            None
        } else {
            self.iter.idx(index + self.n)
        }
    }
}

/// An iterator which only iterates over the first `n` iterations of `iter`.
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Take<T> {
    iter: T,
    n: uint
}

impl<A, T: Iterator<A>> Iterator<A> for Take<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
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

impl<A, T: RandomAccessIterator<A>> RandomAccessIterator<A> for Take<T> {
    #[inline]
    fn indexable(&self) -> uint {
        cmp::min(self.iter.indexable(), self.n)
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
        if index >= self.n {
            None
        } else {
            self.iter.idx(index)
        }
    }
}


/// An iterator to maintain state while iterating another iterator
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Scan<'a, A, B, T, St> {
    iter: T,
    f: |&mut St, A|: 'a -> Option<B>,

    /// The current internal state to be passed to the closure next.
    pub state: St,
}

impl<'a, A, B, T: Iterator<A>, St> Iterator<B> for Scan<'a, A, B, T, St> {
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
pub struct FlatMap<'a, A, T, U> {
    iter: T,
    f: |A|: 'a -> U,
    frontiter: Option<U>,
    backiter: Option<U>,
}

impl<'a, A, T: Iterator<A>, B, U: Iterator<B>> Iterator<B> for FlatMap<'a, A, T, U> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        loop {
            for inner in self.frontiter.mut_iter() {
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
            ((0, Some(0)), Some(a), Some(b)) => (lo, a.checked_add(&b)),
            _ => (lo, None)
        }
    }
}

impl<'a,
     A, T: DoubleEndedIterator<A>,
     B, U: DoubleEndedIterator<B>> DoubleEndedIterator<B>
     for FlatMap<'a, A, T, U> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        loop {
            for inner in self.backiter.mut_iter() {
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
#[deriving(Clone)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Fuse<T> {
    iter: T,
    done: bool
}

impl<A, T: Iterator<A>> Iterator<A> for Fuse<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
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

impl<A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for Fuse<T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
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
impl<A, T: RandomAccessIterator<A>> RandomAccessIterator<A> for Fuse<T> {
    #[inline]
    fn indexable(&self) -> uint {
        self.iter.indexable()
    }

    #[inline]
    fn idx(&mut self, index: uint) -> Option<A> {
        self.iter.idx(index)
    }
}

impl<T> Fuse<T> {
    /// Resets the fuse such that the next call to .next() or .next_back() will
    /// call the underlying iterator again even if it previously returned None.
    #[inline]
    pub fn reset_fuse(&mut self) {
        self.done = false
    }
}

/// An iterator that calls a function with a reference to each
/// element before yielding it.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Inspect<'a, A, T> {
    iter: T,
    f: |&A|: 'a
}

impl<'a, A, T> Inspect<'a, A, T> {
    #[inline]
    fn do_inspect(&mut self, elt: Option<A>) -> Option<A> {
        match elt {
            Some(ref a) => (self.f)(a),
            None => ()
        }

        elt
    }
}

impl<'a, A, T: Iterator<A>> Iterator<A> for Inspect<'a, A, T> {
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

impl<'a, A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A>
for Inspect<'a, A, T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        let next = self.iter.next_back();
        self.do_inspect(next)
    }
}

impl<'a, A, T: RandomAccessIterator<A>> RandomAccessIterator<A>
for Inspect<'a, A, T> {
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

/// An iterator which just modifies the contained state throughout iteration.
pub struct Unfold<'a, A, St> {
    f: |&mut St|: 'a -> Option<A>,
    /// Internal state that will be yielded on the next iteration
    pub state: St,
}

impl<'a, A, St> Unfold<'a, A, St> {
    /// Creates a new iterator with the specified closure as the "iterator
    /// function" and an initial state to eventually pass to the iterator
    #[inline]
    pub fn new<'a>(initial_state: St, f: |&mut St|: 'a -> Option<A>)
               -> Unfold<'a, A, St> {
        Unfold {
            f: f,
            state: initial_state
        }
    }
}

impl<'a, A, St> Iterator<A> for Unfold<'a, A, St> {
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
#[deriving(Clone)]
pub struct Counter<A> {
    /// The current state the counter is at (next value to be yielded)
    state: A,
    /// The amount that this iterator is stepping by
    step: A,
}

/// Creates a new counter with the specified start/step
#[inline]
pub fn count<A>(start: A, step: A) -> Counter<A> {
    Counter{state: start, step: step}
}

impl<A: Add<A, A> + Clone> Iterator<A> for Counter<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let result = self.state.clone();
        self.state = self.state + self.step;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (uint::MAX, None) // Too bad we can't specify an infinite lower bound
    }
}

/// An iterator over the range [start, stop)
#[deriving(Clone)]
pub struct Range<A> {
    state: A,
    stop: A,
    one: A
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
pub fn range<A: Add<A, A> + PartialOrd + Clone + One>(start: A, stop: A) -> Range<A> {
    Range{state: start, stop: stop, one: One::one()}
}

// FIXME: #10414: Unfortunate type bound
impl<A: Add<A, A> + PartialOrd + Clone + ToPrimitive> Iterator<A> for Range<A> {
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
                let sz = self.stop.to_i64().map(|b| b.checked_sub(&a));
                match sz {
                    Some(Some(bound)) => bound.to_uint(),
                    _ => None,
                }
            },
            None => match self.state.to_u64() {
                Some(a) => {
                    let sz = self.stop.to_u64().map(|b| b.checked_sub(&a));
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
impl<A: Int + PartialOrd + Clone + ToPrimitive> DoubleEndedIterator<A> for Range<A> {
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
#[deriving(Clone)]
pub struct RangeInclusive<A> {
    range: Range<A>,
    done: bool,
}

/// Return an iterator over the range [start, stop]
#[inline]
pub fn range_inclusive<A: Add<A, A> + PartialOrd + Clone + One>(start: A, stop: A)
    -> RangeInclusive<A> {
    RangeInclusive{range: range(start, stop), done: false}
}

impl<A: Add<A, A> + PartialOrd + Clone + ToPrimitive> Iterator<A> for RangeInclusive<A> {
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
                Some(x) => x.checked_add(&1),
                None => None
            };
            (lo, hi)
        }
    }
}

impl<A: Sub<A, A> + Int + PartialOrd + Clone + ToPrimitive> DoubleEndedIterator<A>
    for RangeInclusive<A> {
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
#[deriving(Clone)]
pub struct RangeStep<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
}

/// Return an iterator over the range [start, stop) by `step`. It handles overflow by stopping.
#[inline]
pub fn range_step<A: CheckedAdd + PartialOrd +
                  Clone + Zero>(start: A, stop: A, step: A) -> RangeStep<A> {
    let rev = step < Zero::zero();
    RangeStep{state: start, stop: stop, step: step, rev: rev}
}

impl<A: CheckedAdd + PartialOrd + Clone> Iterator<A> for RangeStep<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if (self.rev && self.state > self.stop) || (!self.rev && self.state < self.stop) {
            let result = self.state.clone();
            match self.state.checked_add(&self.step) {
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
#[deriving(Clone)]
pub struct RangeStepInclusive<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
    done: bool,
}

/// Return an iterator over the range [start, stop] by `step`. It handles overflow by stopping.
#[inline]
pub fn range_step_inclusive<A: CheckedAdd + PartialOrd + Clone + Zero>(start: A, stop: A,
                                                                step: A) -> RangeStepInclusive<A> {
    let rev = step < Zero::zero();
    RangeStepInclusive{state: start, stop: stop, step: step, rev: rev, done: false}
}

impl<A: CheckedAdd + PartialOrd + Clone + PartialEq> Iterator<A> for RangeStepInclusive<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if !self.done && ((self.rev && self.state >= self.stop) ||
                          (!self.rev && self.state <= self.stop)) {
            let result = self.state.clone();
            match self.state.checked_add(&self.step) {
                Some(x) => self.state = x,
                None => self.done = true
            }
            Some(result)
        } else {
            None
        }
    }
}

/// An iterator that repeats an element endlessly
#[deriving(Clone)]
pub struct Repeat<A> {
    element: A
}

impl<A: Clone> Repeat<A> {
    /// Create a new `Repeat` that endlessly repeats the element `elt`.
    #[inline]
    pub fn new(elt: A) -> Repeat<A> {
        Repeat{element: elt}
    }
}

impl<A: Clone> Iterator<A> for Repeat<A> {
    #[inline]
    fn next(&mut self) -> Option<A> { self.idx(0) }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { (uint::MAX, None) }
}

impl<A: Clone> DoubleEndedIterator<A> for Repeat<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.idx(0) }
}

impl<A: Clone> RandomAccessIterator<A> for Repeat<A> {
    #[inline]
    fn indexable(&self) -> uint { uint::MAX }
    #[inline]
    fn idx(&mut self, _: uint) -> Option<A> { Some(self.element.clone()) }
}

type IterateState<'a, T> = (|T|: 'a -> T, Option<T>, bool);

/// An iterator that repeatedly applies a given function, starting
/// from a given seed value.
pub type Iterate<'a, T> = Unfold<'a, T, IterateState<'a, T>>;

/// Creates a new iterator that produces an infinite sequence of
/// repeated applications of the given function `f`.
#[allow(visible_private_types)]
pub fn iterate<'a, T: Clone>(f: |T|: 'a -> T, seed: T) -> Iterate<'a, T> {
    Unfold::new((f, Some(seed), true), |st| {
        let &(ref mut f, ref mut val, ref mut first) = st;
        if *first {
            *first = false;
        } else {
            val.mutate(|x| (*f)(x));
        }
        val.clone()
    })
}

/// Functions for lexicographical ordering of sequences.
///
/// Lexicographical ordering through `<`, `<=`, `>=`, `>` requires
/// that the elements implement both `PartialEq` and `PartialOrd`.
///
/// If two sequences are equal up until the point where one ends,
/// the shorter sequence compares less.
pub mod order {
    use cmp;
    use cmp::{Eq, Ord, PartialOrd, PartialEq};
    use option::{Option, Some, None};
    use super::Iterator;

    /// Compare `a` and `b` for equality using `Eq`
    pub fn equals<A: Eq, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (None, _) | (_, None) => return false,
                (Some(x), Some(y)) => if x != y { return false },
            }
        }
    }

    /// Order `a` and `b` lexicographically using `Ord`
    pub fn cmp<A: Ord, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> cmp::Ordering {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return cmp::Equal,
                (None, _   ) => return cmp::Less,
                (_   , None) => return cmp::Greater,
                (Some(x), Some(y)) => match x.cmp(&y) {
                    cmp::Equal => (),
                    non_eq => return non_eq,
                },
            }
        }
    }

    /// Order `a` and `b` lexicographically using `PartialOrd`
    pub fn partial_cmp<A: PartialOrd, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S)
            -> Option<cmp::Ordering> {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return Some(cmp::Equal),
                (None, _   ) => return Some(cmp::Less),
                (_   , None) => return Some(cmp::Greater),
                (Some(x), Some(y)) => match x.partial_cmp(&y) {
                    Some(cmp::Equal) => (),
                    non_eq => return non_eq,
                },
            }
        }
    }

    /// Compare `a` and `b` for equality (Using partial equality, `PartialEq`)
    pub fn eq<A: PartialEq, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return true,
                (None, _) | (_, None) => return false,
                (Some(x), Some(y)) => if !x.eq(&y) { return false },
            }
        }
    }

    /// Compare `a` and `b` for nonequality (Using partial equality, `PartialEq`)
    pub fn ne<A: PartialEq, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
        loop {
            match (a.next(), b.next()) {
                (None, None) => return false,
                (None, _) | (_, None) => return true,
                (Some(x), Some(y)) => if x.ne(&y) { return true },
            }
        }
    }

    /// Return `a` < `b` lexicographically (Using partial order, `PartialOrd`)
    pub fn lt<A: PartialOrd, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
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
    pub fn le<A: PartialOrd, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
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
    pub fn gt<A: PartialOrd, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
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
    pub fn ge<A: PartialOrd, T: Iterator<A>, S: Iterator<A>>(mut a: T, mut b: S) -> bool {
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
