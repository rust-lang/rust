// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Composable external iterators

The `Iterator` trait defines an interface for objects which implement iteration as a state machine.

Algorithms like `zip` are provided as `Iterator` implementations which wrap other objects
implementing the `Iterator` trait.

*/

use cmp;
use iter::Times;
use num::{Zero, One};
use option::{Option, Some, None};
use ops::{Add, Mul};
use cmp::Ord;
use clone::Clone;
use uint;

/// Conversion from an `Iterator`
pub trait FromIterator<A, T: Iterator<A>> {
    /// Build a container with elements from an external iterator.
    fn from_iterator(iterator: &mut T) -> Self;
}

/// A type growable from an `Iterator` implementation
pub trait Extendable<A, T: Iterator<A>>: FromIterator<A, T> {
    /// Extend a container with the elements yielded by an iterator
    fn extend(&mut self, iterator: &mut T);
}

/// An interface for dealing with "external iterators". These types of iterators
/// can be resumed at any time as all state is stored internally as opposed to
/// being located on the call stack.
pub trait Iterator<A> {
    /// Advance the iterator and return the next value. Return `None` when the end is reached.
    fn next(&mut self) -> Option<A>;

    /// Return a lower bound and upper bound on the remaining length of the iterator.
    ///
    /// The common use case for the estimate is pre-allocating space to store the results.
    fn size_hint(&self) -> (uint, Option<uint>) { (0, None) }
}

/// A range iterator able to yield elements from both ends
pub trait DoubleEndedIterator<A>: Iterator<A> {
    /// Yield an element from the end of the range, returning `None` if the range is empty.
    fn next_back(&mut self) -> Option<A>;
}

/// An object implementing random access indexing by `uint`
///
/// A `RandomAccessIterator` should be either infinite or a `DoubleEndedIterator`.
pub trait RandomAccessIterator<A>: Iterator<A> {
    /// Return the number of indexable elements. At most `std::uint::max_value`
    /// elements are indexable, even if the iterator represents a longer range.
    fn indexable(&self) -> uint;

    /// Return an element at an index
    fn idx(&self, index: uint) -> Option<A>;
}

/// Iterator adaptors provided for every `DoubleEndedIterator` implementation.
///
/// In the future these will be default methods instead of a utility trait.
pub trait DoubleEndedIteratorUtil {
    /// Flip the direction of the iterator
    fn invert(self) -> Invert<Self>;
}

/// Iterator adaptors provided for every `DoubleEndedIterator` implementation.
///
/// In the future these will be default methods instead of a utility trait.
impl<A, T: DoubleEndedIterator<A>> DoubleEndedIteratorUtil for T {
    /// Flip the direction of the iterator
    #[inline]
    fn invert(self) -> Invert<T> {
        Invert{iter: self}
    }
}

/// An double-ended iterator with the direction inverted
#[deriving(Clone)]
pub struct Invert<T> {
    priv iter: T
}

impl<A, T: DoubleEndedIterator<A>> Iterator<A> for Invert<T> {
    #[inline]
    fn next(&mut self) -> Option<A> { self.iter.next_back() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

impl<A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for Invert<T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.iter.next() }
}

/// Iterator adaptors provided for every `Iterator` implementation. The adaptor objects are also
/// implementations of the `Iterator` trait.
///
/// In the future these will be default methods instead of a utility trait.
pub trait IteratorUtil<A> {
    /// Chain this iterator with another, returning a new iterator which will
    /// finish iterating over the current iterator, and then it will iterate
    /// over the other specified iterator.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [0];
    /// let b = [1];
    /// let mut it = a.iter().chain_(b.iter());
    /// assert_eq!(it.next().get(), &0);
    /// assert_eq!(it.next().get(), &1);
    /// assert!(it.next().is_none());
    /// ~~~
    fn chain_<U: Iterator<A>>(self, other: U) -> Chain<Self, U>;

    /// Creates an iterator which iterates over both this and the specified
    /// iterators simultaneously, yielding the two elements as pairs. When
    /// either iterator returns None, all further invocations of next() will
    /// return None.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [0];
    /// let b = [1];
    /// let mut it = a.iter().zip(b.iter());
    /// assert_eq!(it.next().get(), (&0, &1));
    /// assert!(it.next().is_none());
    /// ~~~
    fn zip<B, U: Iterator<B>>(self, other: U) -> Zip<Self, U>;

    // FIXME: #5898: should be called map
    /// Creates a new iterator which will apply the specified function to each
    /// element returned by the first, yielding the mapped element instead.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2];
    /// let mut it = a.iter().transform(|&x| 2 * x);
    /// assert_eq!(it.next().get(), 2);
    /// assert_eq!(it.next().get(), 4);
    /// assert!(it.next().is_none());
    /// ~~~
    fn transform<'r, B>(self, f: &'r fn(A) -> B) -> Map<'r, A, B, Self>;

    /// Creates an iterator which applies the predicate to each element returned
    /// by this iterator. Only elements which have the predicate evaluate to
    /// `true` will be yielded.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2];
    /// let mut it = a.iter().filter(|&x| *x > 1);
    /// assert_eq!(it.next().get(), &2);
    /// assert!(it.next().is_none());
    /// ~~~
    fn filter<'r>(self, predicate: &'r fn(&A) -> bool) -> Filter<'r, A, Self>;

    /// Creates an iterator which both filters and maps elements.
    /// If the specified function returns None, the element is skipped.
    /// Otherwise the option is unwrapped and the new value is yielded.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2];
    /// let mut it = a.iter().filter_map(|&x| if x > 1 {Some(2 * x)} else {None});
    /// assert_eq!(it.next().get(), 4);
    /// assert!(it.next().is_none());
    /// ~~~
    fn filter_map<'r,  B>(self, f: &'r fn(A) -> Option<B>) -> FilterMap<'r, A, B, Self>;

    /// Creates an iterator which yields a pair of the value returned by this
    /// iterator plus the current index of iteration.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [100, 200];
    /// let mut it = a.iter().enumerate();
    /// assert_eq!(it.next().get(), (0, &100));
    /// assert_eq!(it.next().get(), (1, &200));
    /// assert!(it.next().is_none());
    /// ~~~
    fn enumerate(self) -> Enumerate<Self>;

    /// Creates an iterator which invokes the predicate on elements until it
    /// returns false. Once the predicate returns false, all further elements are
    /// yielded.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 2, 1];
    /// let mut it = a.iter().skip_while(|&a| *a < 3);
    /// assert_eq!(it.next().get(), &3);
    /// assert_eq!(it.next().get(), &2);
    /// assert_eq!(it.next().get(), &1);
    /// assert!(it.next().is_none());
    /// ~~~
    fn skip_while<'r>(self, predicate: &'r fn(&A) -> bool) -> SkipWhile<'r, A, Self>;

    /// Creates an iterator which yields elements so long as the predicate
    /// returns true. After the predicate returns false for the first time, no
    /// further elements will be yielded.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 2, 1];
    /// let mut it = a.iter().take_while(|&a| *a < 3);
    /// assert_eq!(it.next().get(), &1);
    /// assert_eq!(it.next().get(), &2);
    /// assert!(it.next().is_none());
    /// ~~~
    fn take_while<'r>(self, predicate: &'r fn(&A) -> bool) -> TakeWhile<'r, A, Self>;

    /// Creates an iterator which skips the first `n` elements of this iterator,
    /// and then it yields all further items.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().skip(3);
    /// assert_eq!(it.next().get(), &4);
    /// assert_eq!(it.next().get(), &5);
    /// assert!(it.next().is_none());
    /// ~~~
    fn skip(self, n: uint) -> Skip<Self>;

    // FIXME: #5898: should be called take
    /// Creates an iterator which yields the first `n` elements of this
    /// iterator, and then it will always return None.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().take_(3);
    /// assert_eq!(it.next().get(), &1);
    /// assert_eq!(it.next().get(), &2);
    /// assert_eq!(it.next().get(), &3);
    /// assert!(it.next().is_none());
    /// ~~~
    fn take_(self, n: uint) -> Take<Self>;

    /// Creates a new iterator which behaves in a similar fashion to foldl.
    /// There is a state which is passed between each iteration and can be
    /// mutated as necessary. The yielded values from the closure are yielded
    /// from the Scan instance when not None.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().scan(1, |fac, &x| {
    ///   *fac = *fac * x;
    ///   Some(*fac)
    /// });
    /// assert_eq!(it.next().get(), 1);
    /// assert_eq!(it.next().get(), 2);
    /// assert_eq!(it.next().get(), 6);
    /// assert_eq!(it.next().get(), 24);
    /// assert_eq!(it.next().get(), 120);
    /// assert!(it.next().is_none());
    /// ~~~
    fn scan<'r, St, B>(self, initial_state: St, f: &'r fn(&mut St, A) -> Option<B>)
        -> Scan<'r, A, B, Self, St>;

    /// Creates an iterator that maps each element to an iterator,
    /// and yields the elements of the produced iterators
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let xs = [2u, 3];
    /// let ys = [0u, 1, 0, 1, 2];
    /// let mut it = xs.iter().flat_map_(|&x| Counter::new(0u, 1).take_(x));
    /// // Check that `it` has the same elements as `ys`
    /// let mut i = 0;
    /// for it.advance |x: uint| {
    ///     assert_eq!(x, ys[i]);
    ///     i += 1;
    /// }
    /// ~~~
    // FIXME: #5898: should be called `flat_map`
    fn flat_map_<'r, B, U: Iterator<B>>(self, f: &'r fn(A) -> U)
        -> FlatMap<'r, A, Self, U>;

    /// Creates an iterator that calls a function with a reference to each
    /// element before yielding it. This is often useful for debugging an
    /// iterator pipeline.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    ///let xs = [1u, 4, 2, 3, 8, 9, 6];
    ///let sum = xs.iter()
    ///            .transform(|&x| x)
    ///            .peek_(|&x| debug!("filtering %u", x))
    ///            .filter(|&x| x % 2 == 0)
    ///            .peek_(|&x| debug!("%u made it through", x))
    ///            .sum();
    ///println(sum.to_str());
    /// ~~~
    // FIXME: #5898: should be called `peek`
    fn peek_<'r>(self, f: &'r fn(&A)) -> Peek<'r, A, Self>;

    /// An adaptation of an external iterator to the for-loop protocol of rust.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// use std::iterator::Counter;
    ///
    /// for Counter::new(0, 10).advance |i| {
    ///     printfln!("%d", i);
    /// }
    /// ~~~
    fn advance(&mut self, f: &fn(A) -> bool) -> bool;

    /// Loops through the entire iterator, collecting all of the elements into
    /// a container implementing `FromIterator`.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let b: ~[int] = a.iter().transform(|&x| x).collect();
    /// assert!(a == b);
    /// ~~~
    fn collect<B: FromIterator<A, Self>>(&mut self) -> B;

    /// Loops through the entire iterator, collecting all of the elements into
    /// a unique vector. This is simply collect() specialized for vectors.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let b: ~[int] = a.iter().transform(|&x| x).to_owned_vec();
    /// assert!(a == b);
    /// ~~~
    fn to_owned_vec(&mut self) -> ~[A];

    /// Loops through `n` iterations, returning the `n`th element of the
    /// iterator.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.nth(2).get() == &3);
    /// assert!(it.nth(2) == None);
    /// ~~~
    fn nth(&mut self, n: uint) -> Option<A>;

    /// Loops through the entire iterator, returning the last element of the
    /// iterator.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().last().get() == &5);
    /// ~~~
    // FIXME: #5898: should be called `last`
    fn last_(&mut self) -> Option<A>;

    /// Performs a fold operation over the entire iterator, returning the
    /// eventual state at the end of the iteration.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().fold(0, |a, &b| a + b) == 15);
    /// ~~~
    fn fold<B>(&mut self, start: B, f: &fn(B, A) -> B) -> B;

    // FIXME: #5898: should be called len
    /// Counts the number of elements in this iterator.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.len_() == 5);
    /// assert!(it.len_() == 0);
    /// ~~~
    fn len_(&mut self) -> uint;

    /// Tests whether the predicate holds true for all elements in the iterator.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().all(|&x| *x > 0));
    /// assert!(!a.iter().all(|&x| *x > 2));
    /// ~~~
    fn all(&mut self, f: &fn(A) -> bool) -> bool;

    /// Tests whether any element of an iterator satisfies the specified
    /// predicate.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter();
    /// assert!(it.any(|&x| *x == 3));
    /// assert!(!it.any(|&x| *x == 3));
    /// ~~~
    fn any(&mut self, f: &fn(A) -> bool) -> bool;

    /// Return the first element satisfying the specified predicate
    fn find_(&mut self, predicate: &fn(&A) -> bool) -> Option<A>;

    /// Return the index of the first element satisfying the specified predicate
    fn position(&mut self, predicate: &fn(A) -> bool) -> Option<uint>;

    /// Count the number of elements satisfying the specified predicate
    fn count(&mut self, predicate: &fn(A) -> bool) -> uint;

    /// Return the element that gives the maximum value from the specfied function
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let xs = [-3, 0, 1, 5, -10];
    /// assert_eq!(*xs.iter().max_by(|x| x.abs()).unwrap(), -10);
    /// ~~~
    fn max_by<B: Ord>(&mut self, f: &fn(&A) -> B) -> Option<A>;

    /// Return the element that gives the minimum value from the specfied function
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let xs = [-3, 0, 1, 5, -10];
    /// assert_eq!(*xs.iter().min_by(|x| x.abs()).unwrap(), 0);
    /// ~~~
    fn min_by<B: Ord>(&mut self, f: &fn(&A) -> B) -> Option<A>;
}

/// Iterator adaptors provided for every `Iterator` implementation. The adaptor objects are also
/// implementations of the `Iterator` trait.
///
/// In the future these will be default methods instead of a utility trait.
impl<A, T: Iterator<A>> IteratorUtil<A> for T {
    #[inline]
    fn chain_<U: Iterator<A>>(self, other: U) -> Chain<T, U> {
        Chain{a: self, b: other, flag: false}
    }

    #[inline]
    fn zip<B, U: Iterator<B>>(self, other: U) -> Zip<T, U> {
        Zip{a: self, b: other}
    }

    // FIXME: #5898: should be called map
    #[inline]
    fn transform<'r, B>(self, f: &'r fn(A) -> B) -> Map<'r, A, B, T> {
        Map{iter: self, f: f}
    }

    #[inline]
    fn filter<'r>(self, predicate: &'r fn(&A) -> bool) -> Filter<'r, A, T> {
        Filter{iter: self, predicate: predicate}
    }

    #[inline]
    fn filter_map<'r, B>(self, f: &'r fn(A) -> Option<B>) -> FilterMap<'r, A, B, T> {
        FilterMap { iter: self, f: f }
    }

    #[inline]
    fn enumerate(self) -> Enumerate<T> {
        Enumerate{iter: self, count: 0}
    }

    #[inline]
    fn skip_while<'r>(self, predicate: &'r fn(&A) -> bool) -> SkipWhile<'r, A, T> {
        SkipWhile{iter: self, flag: false, predicate: predicate}
    }

    #[inline]
    fn take_while<'r>(self, predicate: &'r fn(&A) -> bool) -> TakeWhile<'r, A, T> {
        TakeWhile{iter: self, flag: false, predicate: predicate}
    }

    #[inline]
    fn skip(self, n: uint) -> Skip<T> {
        Skip{iter: self, n: n}
    }

    // FIXME: #5898: should be called take
    #[inline]
    fn take_(self, n: uint) -> Take<T> {
        Take{iter: self, n: n}
    }

    #[inline]
    fn scan<'r, St, B>(self, initial_state: St, f: &'r fn(&mut St, A) -> Option<B>)
        -> Scan<'r, A, B, T, St> {
        Scan{iter: self, f: f, state: initial_state}
    }

    #[inline]
    fn flat_map_<'r, B, U: Iterator<B>>(self, f: &'r fn(A) -> U)
        -> FlatMap<'r, A, T, U> {
        FlatMap{iter: self, f: f, subiter: None }
    }

    // FIXME: #5898: should be called `peek`
    #[inline]
    fn peek_<'r>(self, f: &'r fn(&A)) -> Peek<'r, A, T> {
        Peek{iter: self, f: f}
    }

    /// A shim implementing the `for` loop iteration protocol for iterator objects
    #[inline]
    fn advance(&mut self, f: &fn(A) -> bool) -> bool {
        loop {
            match self.next() {
                Some(x) => {
                    if !f(x) { return false; }
                }
                None => { return true; }
            }
        }
    }

    #[inline]
    fn collect<B: FromIterator<A, T>>(&mut self) -> B {
        FromIterator::from_iterator(self)
    }

    #[inline]
    fn to_owned_vec(&mut self) -> ~[A] {
        self.collect()
    }

    /// Return the `n`th item yielded by an iterator.
    #[inline]
    fn nth(&mut self, mut n: uint) -> Option<A> {
        loop {
            match self.next() {
                Some(x) => if n == 0 { return Some(x) },
                None => return None
            }
            n -= 1;
        }
    }

    /// Return the last item yielded by an iterator.
    #[inline]
    fn last_(&mut self) -> Option<A> {
        let mut last = None;
        for self.advance |x| { last = Some(x); }
        last
    }

    /// Reduce an iterator to an accumulated value
    #[inline]
    fn fold<B>(&mut self, init: B, f: &fn(B, A) -> B) -> B {
        let mut accum = init;
        loop {
            match self.next() {
                Some(x) => { accum = f(accum, x); }
                None    => { break; }
            }
        }
        accum
    }

    /// Count the number of items yielded by an iterator
    #[inline]
    fn len_(&mut self) -> uint { self.fold(0, |cnt, _x| cnt + 1) }

    #[inline]
    fn all(&mut self, f: &fn(A) -> bool) -> bool {
        for self.advance |x| { if !f(x) { return false; } }
        true
    }

    #[inline]
    fn any(&mut self, f: &fn(A) -> bool) -> bool {
        for self.advance |x| { if f(x) { return true; } }
        false
    }

    /// Return the first element satisfying the specified predicate
    #[inline]
    fn find_(&mut self, predicate: &fn(&A) -> bool) -> Option<A> {
        for self.advance |x| {
            if predicate(&x) { return Some(x) }
        }
        None
    }

    /// Return the index of the first element satisfying the specified predicate
    #[inline]
    fn position(&mut self, predicate: &fn(A) -> bool) -> Option<uint> {
        let mut i = 0;
        for self.advance |x| {
            if predicate(x) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[inline]
    fn count(&mut self, predicate: &fn(A) -> bool) -> uint {
        let mut i = 0;
        for self.advance |x| {
            if predicate(x) { i += 1 }
        }
        i
    }

    #[inline]
    fn max_by<B: Ord>(&mut self, f: &fn(&A) -> B) -> Option<A> {
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
        }).map_consume(|(x, _)| x)
    }

    #[inline]
    fn min_by<B: Ord>(&mut self, f: &fn(&A) -> B) -> Option<A> {
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
        }).map_consume(|(x, _)| x)
    }
}

/// A trait for iterators over elements which can be added together
pub trait AdditiveIterator<A> {
    /// Iterates over the entire iterator, summing up all the elements
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// let mut it = a.iter().transform(|&x| x);
    /// assert!(it.sum() == 15);
    /// ~~~
    fn sum(&mut self) -> A;
}

impl<A: Add<A, A> + Zero, T: Iterator<A>> AdditiveIterator<A> for T {
    #[inline]
    fn sum(&mut self) -> A { self.fold(Zero::zero::<A>(), |s, x| s + x) }
}

/// A trait for iterators over elements whose elements can be multiplied
/// together.
pub trait MultiplicativeIterator<A> {
    /// Iterates over the entire iterator, multiplying all the elements
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// use std::iterator::Counter;
    ///
    /// fn factorial(n: uint) -> uint {
    ///     Counter::new(1u, 1).take_while(|&i| i <= n).product()
    /// }
    /// assert!(factorial(0) == 1);
    /// assert!(factorial(1) == 1);
    /// assert!(factorial(5) == 120);
    /// ~~~
    fn product(&mut self) -> A;
}

impl<A: Mul<A, A> + One, T: Iterator<A>> MultiplicativeIterator<A> for T {
    #[inline]
    fn product(&mut self) -> A { self.fold(One::one::<A>(), |p, x| p * x) }
}

/// A trait for iterators over elements which can be compared to one another.
/// The type of each element must ascribe to the `Ord` trait.
pub trait OrdIterator<A> {
    /// Consumes the entire iterator to return the maximum element.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().max().get() == &5);
    /// ~~~
    fn max(&mut self) -> Option<A>;

    /// Consumes the entire iterator to return the minimum element.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = [1, 2, 3, 4, 5];
    /// assert!(a.iter().min().get() == &1);
    /// ~~~
    fn min(&mut self) -> Option<A>;
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
}

/// A trait for iterators that are clonable.
pub trait ClonableIterator {
    /// Repeats an iterator endlessly
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let a = Counter::new(1,1).take_(1);
    /// let mut cy = a.cycle();
    /// assert_eq!(cy.next(), Some(1));
    /// assert_eq!(cy.next(), Some(1));
    /// ~~~
    fn cycle(self) -> Cycle<Self>;
}

impl<A, T: Clone + Iterator<A>> ClonableIterator for T {
    #[inline]
    fn cycle(self) -> Cycle<T> {
        Cycle{orig: self.clone(), iter: self}
    }
}

/// An iterator that repeats endlessly
#[deriving(Clone)]
pub struct Cycle<T> {
    priv orig: T,
    priv iter: T,
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
            _ => (uint::max_value, None)
        }
    }
}

/// An iterator which strings two iterators together
#[deriving(Clone)]
pub struct Chain<T, U> {
    priv a: T,
    priv b: U,
    priv flag: bool
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

        let lower = if uint::max_value - a_lower < b_lower {
            uint::max_value
        } else {
            a_lower + b_lower
        };

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) if uint::max_value - x < y => Some(uint::max_value),
            (Some(x), Some(y)) => Some(x + y),
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
        let total = a + b;
        if total < a || total < b {
            uint::max_value
        } else {
            total
        }
    }

    #[inline]
    fn idx(&self, index: uint) -> Option<A> {
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
pub struct Zip<T, U> {
    priv a: T,
    priv b: U
}

impl<A, B, T: Iterator<A>, U: Iterator<B>> Iterator<(A, B)> for Zip<T, U> {
    #[inline]
    fn next(&mut self) -> Option<(A, B)> {
        match (self.a.next(), self.b.next()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None
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

/// An iterator which maps the values of `iter` with `f`
pub struct Map<'self, A, B, T> {
    priv iter: T,
    priv f: &'self fn(A) -> B
}

impl<'self, A, B, T: Iterator<A>> Iterator<B> for Map<'self, A, B, T> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        match self.iter.next() {
            Some(a) => Some((self.f)(a)),
            _ => None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<'self, A, B, T: DoubleEndedIterator<A>> DoubleEndedIterator<B>
for Map<'self, A, B, T> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        match self.iter.next_back() {
            Some(a) => Some((self.f)(a)),
            _ => None
        }
    }
}

/// An iterator which filters the elements of `iter` with `predicate`
pub struct Filter<'self, A, T> {
    priv iter: T,
    priv predicate: &'self fn(&A) -> bool
}

impl<'self, A, T: Iterator<A>> Iterator<A> for Filter<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        for self.iter.advance |x| {
            if (self.predicate)(&x) {
                return Some(x);
            } else {
                loop
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

impl<'self, A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for Filter<'self, A, T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        loop {
            match self.iter.next_back() {
                None => return None,
                Some(x) => {
                    if (self.predicate)(&x) {
                        return Some(x);
                    } else {
                        loop
                    }
                }
            }
        }
    }
}

/// An iterator which uses `f` to both filter and map elements from `iter`
pub struct FilterMap<'self, A, B, T> {
    priv iter: T,
    priv f: &'self fn(A) -> Option<B>
}

impl<'self, A, B, T: Iterator<A>> Iterator<B> for FilterMap<'self, A, B, T> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        for self.iter.advance |x| {
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

impl<'self, A, B, T: DoubleEndedIterator<A>> DoubleEndedIterator<B>
for FilterMap<'self, A, B, T> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        loop {
            match self.iter.next_back() {
                None => return None,
                Some(x) => {
                    match (self.f)(x) {
                        Some(y) => return Some(y),
                        None => ()
                    }
                }
            }
        }
    }
}

/// An iterator which yields the current count and the element during iteration
#[deriving(Clone)]
pub struct Enumerate<T> {
    priv iter: T,
    priv count: uint
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

/// An iterator which rejects elements while `predicate` is true
pub struct SkipWhile<'self, A, T> {
    priv iter: T,
    priv flag: bool,
    priv predicate: &'self fn(&A) -> bool
}

impl<'self, A, T: Iterator<A>> Iterator<A> for SkipWhile<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut next = self.iter.next();
        if self.flag {
            next
        } else {
            loop {
                match next {
                    Some(x) => {
                        if (self.predicate)(&x) {
                            next = self.iter.next();
                            loop
                        } else {
                            self.flag = true;
                            return Some(x)
                        }
                    }
                    None => return None
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}

/// An iterator which only accepts elements while `predicate` is true
pub struct TakeWhile<'self, A, T> {
    priv iter: T,
    priv flag: bool,
    priv predicate: &'self fn(&A) -> bool
}

impl<'self, A, T: Iterator<A>> Iterator<A> for TakeWhile<'self, A, T> {
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
pub struct Skip<T> {
    priv iter: T,
    priv n: uint
}

impl<A, T: Iterator<A>> Iterator<A> for Skip<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut next = self.iter.next();
        if self.n == 0 {
            next
        } else {
            let n = self.n;
            for n.times {
                match next {
                    Some(_) => {
                        next = self.iter.next();
                        loop
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

        let lower = if lower >= self.n { lower - self.n } else { 0 };

        let upper = match upper {
            Some(x) if x >= self.n => Some(x - self.n),
            Some(_) => Some(0),
            None => None
        };

        (lower, upper)
    }
}

/// An iterator which only iterates over the first `n` iterations of `iter`.
#[deriving(Clone)]
pub struct Take<T> {
    priv iter: T,
    priv n: uint
}

impl<A, T: Iterator<A>> Iterator<A> for Take<T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let next = self.iter.next();
        if self.n != 0 {
            self.n -= 1;
            next
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

/// An iterator to maintain state while iterating another iterator
pub struct Scan<'self, A, B, T, St> {
    priv iter: T,
    priv f: &'self fn(&mut St, A) -> Option<B>,

    /// The current internal state to be passed to the closure next.
    state: St
}

impl<'self, A, B, T: Iterator<A>, St> Iterator<B> for Scan<'self, A, B, T, St> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().chain(|a| (self.f)(&mut self.state, a))
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
pub struct FlatMap<'self, A, T, U> {
    priv iter: T,
    priv f: &'self fn(A) -> U,
    priv subiter: Option<U>,
}

impl<'self, A, T: Iterator<A>, B, U: Iterator<B>> Iterator<B> for
    FlatMap<'self, A, T, U> {
    #[inline]
    fn next(&mut self) -> Option<B> {
        loop {
            for self.subiter.mut_iter().advance |inner| {
                for inner.advance |x| {
                    return Some(x)
                }
            }
            match self.iter.next().map_consume(|x| (self.f)(x)) {
                None => return None,
                next => self.subiter = next,
            }
        }
    }
}

/// An iterator that calls a function with a reference to each
/// element before yielding it.
pub struct Peek<'self, A, T> {
    priv iter: T,
    priv f: &'self fn(&A)
}

impl<'self, A, T: Iterator<A>> Iterator<A> for Peek<'self, A, T> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let next = self.iter.next();

        match next {
            Some(ref a) => (self.f)(a),
            None => ()
        }

        next
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<'self, A, T: DoubleEndedIterator<A>> DoubleEndedIterator<A> for Peek<'self, A, T> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        let next = self.iter.next_back();

        match next {
            Some(ref a) => (self.f)(a),
            None => ()
        }

        next
    }
}

/// An iterator which just modifies the contained state throughout iteration.
pub struct Unfoldr<'self, A, St> {
    priv f: &'self fn(&mut St) -> Option<A>,
    /// Internal state that will be yielded on the next iteration
    state: St
}

impl<'self, A, St> Unfoldr<'self, A, St> {
    /// Creates a new iterator with the specified closure as the "iterator
    /// function" and an initial state to eventually pass to the iterator
    #[inline]
    pub fn new<'a>(initial_state: St, f: &'a fn(&mut St) -> Option<A>)
        -> Unfoldr<'a, A, St> {
        Unfoldr {
            f: f,
            state: initial_state
        }
    }
}

impl<'self, A, St> Iterator<A> for Unfoldr<'self, A, St> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        (self.f)(&mut self.state)
    }
}

/// An infinite iterator starting at `start` and advancing by `step` with each
/// iteration
#[deriving(Clone)]
pub struct Counter<A> {
    /// The current state the counter is at (next value to be yielded)
    state: A,
    /// The amount that this iterator is stepping by
    step: A
}

impl<A> Counter<A> {
    /// Creates a new counter with the specified start/step
    #[inline]
    pub fn new(start: A, step: A) -> Counter<A> {
        Counter{state: start, step: step}
    }
}

impl<A: Add<A, A> + Clone> Iterator<A> for Counter<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        let result = self.state.clone();
        self.state = self.state.add(&self.step); // FIXME: #6050
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (uint::max_value, None) // Too bad we can't specify an infinite lower bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    use uint;

    #[test]
    fn test_counter_from_iter() {
        let mut it = Counter::new(0, 5).take_(10);
        let xs: ~[int] = FromIterator::from_iterator(&mut it);
        assert_eq!(xs, ~[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]);
    }

    #[test]
    fn test_iterator_chain() {
        let xs = [0u, 1, 2, 3, 4, 5];
        let ys = [30u, 40, 50, 60];
        let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
        let mut it = xs.iter().chain_(ys.iter());
        let mut i = 0;
        for it.advance |&x| {
            assert_eq!(x, expected[i]);
            i += 1;
        }
        assert_eq!(i, expected.len());

        let ys = Counter::new(30u, 10).take_(4);
        let mut it = xs.iter().transform(|&x| x).chain_(ys);
        let mut i = 0;
        for it.advance |x| {
            assert_eq!(x, expected[i]);
            i += 1;
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_filter_map() {
        let mut it = Counter::new(0u, 1u).take_(10)
            .filter_map(|x| if x.is_even() { Some(x*x) } else { None });
        assert_eq!(it.collect::<~[uint]>(), ~[0*0, 2*2, 4*4, 6*6, 8*8]);
    }

    #[test]
    fn test_iterator_enumerate() {
        let xs = [0u, 1, 2, 3, 4, 5];
        let mut it = xs.iter().enumerate();
        for it.advance |(i, &x)| {
            assert_eq!(i, x);
        }
    }

    #[test]
    fn test_iterator_take_while() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
        let ys = [0u, 1, 2, 3, 5, 13];
        let mut it = xs.iter().take_while(|&x| *x < 15u);
        let mut i = 0;
        for it.advance |&x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_skip_while() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
        let ys = [15, 16, 17, 19];
        let mut it = xs.iter().skip_while(|&x| *x < 15u);
        let mut i = 0;
        for it.advance |&x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_skip() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
        let ys = [13, 15, 16, 17, 19, 20, 30];
        let mut it = xs.iter().skip(5);
        let mut i = 0;
        for it.advance |&x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_take() {
        let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
        let ys = [0u, 1, 2, 3, 5];
        let mut it = xs.iter().take_(5);
        let mut i = 0;
        for it.advance |&x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_scan() {
        // test the type inference
        fn add(old: &mut int, new: &uint) -> Option<float> {
            *old += *new as int;
            Some(*old as float)
        }
        let xs = [0u, 1, 2, 3, 4];
        let ys = [0f, 1f, 3f, 6f, 10f];

        let mut it = xs.iter().scan(0, add);
        let mut i = 0;
        for it.advance |x| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_iterator_flat_map() {
        let xs = [0u, 3, 6];
        let ys = [0u, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut it = xs.iter().flat_map_(|&x| Counter::new(x, 1).take_(3));
        let mut i = 0;
        for it.advance |x: uint| {
            assert_eq!(x, ys[i]);
            i += 1;
        }
        assert_eq!(i, ys.len());
    }

    #[test]
    fn test_peek() {
        let xs = [1u, 2, 3, 4];
        let mut n = 0;

        let ys = xs.iter()
                   .transform(|&x| x)
                   .peek_(|_| n += 1)
                   .collect::<~[uint]>();

        assert_eq!(n, xs.len());
        assert_eq!(xs, ys.as_slice());
    }

    #[test]
    fn test_unfoldr() {
        fn count(st: &mut uint) -> Option<uint> {
            if *st < 10 {
                let ret = Some(*st);
                *st += 1;
                ret
            } else {
                None
            }
        }

        let mut it = Unfoldr::new(0, count);
        let mut i = 0;
        for it.advance |counted| {
            assert_eq!(counted, i);
            i += 1;
        }
        assert_eq!(i, 10);
    }

    #[test]
    fn test_cycle() {
        let cycle_len = 3;
        let it = Counter::new(0u,1).take_(cycle_len).cycle();
        assert_eq!(it.size_hint(), (uint::max_value, None));
        for it.take_(100).enumerate().advance |(i, x)| {
            assert_eq!(i % cycle_len, x);
        }

        let mut it = Counter::new(0u,1).take_(0).cycle();
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_iterator_nth() {
        let v = &[0, 1, 2, 3, 4];
        for uint::range(0, v.len()) |i| {
            assert_eq!(v.iter().nth(i).unwrap(), &v[i]);
        }
    }

    #[test]
    fn test_iterator_last() {
        let v = &[0, 1, 2, 3, 4];
        assert_eq!(v.iter().last_().unwrap(), &4);
        assert_eq!(v.slice(0, 1).iter().last_().unwrap(), &0);
    }

    #[test]
    fn test_iterator_len() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().len_(), 4);
        assert_eq!(v.slice(0, 10).iter().len_(), 10);
        assert_eq!(v.slice(0, 0).iter().len_(), 0);
    }

    #[test]
    fn test_iterator_sum() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).sum(), 6);
        assert_eq!(v.iter().transform(|&x| x).sum(), 55);
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).sum(), 0);
    }

    #[test]
    fn test_iterator_product() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).product(), 0);
        assert_eq!(v.slice(1, 5).iter().transform(|&x| x).product(), 24);
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).product(), 1);
    }

    #[test]
    fn test_iterator_max() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).max(), Some(3));
        assert_eq!(v.iter().transform(|&x| x).max(), Some(10));
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).max(), None);
    }

    #[test]
    fn test_iterator_min() {
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(v.slice(0, 4).iter().transform(|&x| x).min(), Some(0));
        assert_eq!(v.iter().transform(|&x| x).min(), Some(0));
        assert_eq!(v.slice(0, 0).iter().transform(|&x| x).min(), None);
    }

    #[test]
    fn test_iterator_size_hint() {
        let c = Counter::new(0, 1);
        let v = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let v2 = &[10, 11, 12];
        let vi = v.iter();

        assert_eq!(c.size_hint(), (uint::max_value, None));
        assert_eq!(vi.size_hint(), (10, Some(10)));

        assert_eq!(c.take_(5).size_hint(), (5, Some(5)));
        assert_eq!(c.skip(5).size_hint().second(), None);
        assert_eq!(c.take_while(|_| false).size_hint(), (0, None));
        assert_eq!(c.skip_while(|_| false).size_hint(), (0, None));
        assert_eq!(c.enumerate().size_hint(), (uint::max_value, None));
        assert_eq!(c.chain_(vi.transform(|&i| i)).size_hint(), (uint::max_value, None));
        assert_eq!(c.zip(vi).size_hint(), (10, Some(10)));
        assert_eq!(c.scan(0, |_,_| Some(0)).size_hint(), (0, None));
        assert_eq!(c.filter(|_| false).size_hint(), (0, None));
        assert_eq!(c.transform(|_| 0).size_hint(), (uint::max_value, None));
        assert_eq!(c.filter_map(|_| Some(0)).size_hint(), (0, None));

        assert_eq!(vi.take_(5).size_hint(), (5, Some(5)));
        assert_eq!(vi.take_(12).size_hint(), (10, Some(10)));
        assert_eq!(vi.skip(3).size_hint(), (7, Some(7)));
        assert_eq!(vi.skip(12).size_hint(), (0, Some(0)));
        assert_eq!(vi.take_while(|_| false).size_hint(), (0, Some(10)));
        assert_eq!(vi.skip_while(|_| false).size_hint(), (0, Some(10)));
        assert_eq!(vi.enumerate().size_hint(), (10, Some(10)));
        assert_eq!(vi.chain_(v2.iter()).size_hint(), (13, Some(13)));
        assert_eq!(vi.zip(v2.iter()).size_hint(), (3, Some(3)));
        assert_eq!(vi.scan(0, |_,_| Some(0)).size_hint(), (0, Some(10)));
        assert_eq!(vi.filter(|_| false).size_hint(), (0, Some(10)));
        assert_eq!(vi.transform(|i| i+1).size_hint(), (10, Some(10)));
        assert_eq!(vi.filter_map(|_| Some(0)).size_hint(), (0, Some(10)));
    }

    #[test]
    fn test_collect() {
        let a = ~[1, 2, 3, 4, 5];
        let b: ~[int] = a.iter().transform(|&x| x).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_all() {
        let v = ~&[1, 2, 3, 4, 5];
        assert!(v.iter().all(|&x| x < 10));
        assert!(!v.iter().all(|&x| x.is_even()));
        assert!(!v.iter().all(|&x| x > 100));
        assert!(v.slice(0, 0).iter().all(|_| fail!()));
    }

    #[test]
    fn test_any() {
        let v = ~&[1, 2, 3, 4, 5];
        assert!(v.iter().any(|&x| x < 10));
        assert!(v.iter().any(|&x| x.is_even()));
        assert!(!v.iter().any(|&x| x > 100));
        assert!(!v.slice(0, 0).iter().any(|_| fail!()));
    }

    #[test]
    fn test_find() {
        let v = &[1, 3, 9, 27, 103, 14, 11];
        assert_eq!(*v.iter().find_(|x| *x & 1 == 0).unwrap(), 14);
        assert_eq!(*v.iter().find_(|x| *x % 3 == 0).unwrap(), 3);
        assert!(v.iter().find_(|x| *x % 12 == 0).is_none());
    }

    #[test]
    fn test_position() {
        let v = &[1, 3, 9, 27, 103, 14, 11];
        assert_eq!(v.iter().position(|x| *x & 1 == 0).unwrap(), 5);
        assert_eq!(v.iter().position(|x| *x % 3 == 0).unwrap(), 1);
        assert!(v.iter().position(|x| *x % 12 == 0).is_none());
    }

    #[test]
    fn test_count() {
        let xs = &[1, 2, 2, 1, 5, 9, 0, 2];
        assert_eq!(xs.iter().count(|x| *x == 2), 3);
        assert_eq!(xs.iter().count(|x| *x == 5), 1);
        assert_eq!(xs.iter().count(|x| *x == 95), 0);
    }

    #[test]
    fn test_max_by() {
        let xs = [-3, 0, 1, 5, -10];
        assert_eq!(*xs.iter().max_by(|x| x.abs()).unwrap(), -10);
    }

    #[test]
    fn test_min_by() {
        let xs = [-3, 0, 1, 5, -10];
        assert_eq!(*xs.iter().min_by(|x| x.abs()).unwrap(), 0);
    }

    #[test]
    fn test_invert() {
        let xs = [2, 4, 6, 8, 10, 12, 14, 16];
        let mut it = xs.iter();
        it.next();
        it.next();
        assert_eq!(it.invert().transform(|&x| x).collect::<~[int]>(), ~[16, 14, 12, 10, 8, 6]);
    }

    #[test]
    fn test_double_ended_map() {
        let xs = [1, 2, 3, 4, 5, 6];
        let mut it = xs.iter().transform(|&x| x * -1);
        assert_eq!(it.next(), Some(-1));
        assert_eq!(it.next(), Some(-2));
        assert_eq!(it.next_back(), Some(-6));
        assert_eq!(it.next_back(), Some(-5));
        assert_eq!(it.next(), Some(-3));
        assert_eq!(it.next_back(), Some(-4));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_double_ended_filter() {
        let xs = [1, 2, 3, 4, 5, 6];
        let mut it = xs.iter().filter(|&x| *x & 1 == 0);
        assert_eq!(it.next_back().unwrap(), &6);
        assert_eq!(it.next_back().unwrap(), &4);
        assert_eq!(it.next().unwrap(), &2);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_double_ended_filter_map() {
        let xs = [1, 2, 3, 4, 5, 6];
        let mut it = xs.iter().filter_map(|&x| if x & 1 == 0 { Some(x * 2) } else { None });
        assert_eq!(it.next_back().unwrap(), 12);
        assert_eq!(it.next_back().unwrap(), 8);
        assert_eq!(it.next().unwrap(), 4);
        assert_eq!(it.next_back(), None);
    }

    #[test]
    fn test_double_ended_chain() {
        let xs = [1, 2, 3, 4, 5];
        let ys = ~[7, 9, 11];
        let mut it = xs.iter().chain_(ys.iter()).invert();
        assert_eq!(it.next().unwrap(), &11)
        assert_eq!(it.next().unwrap(), &9)
        assert_eq!(it.next_back().unwrap(), &1)
        assert_eq!(it.next_back().unwrap(), &2)
        assert_eq!(it.next_back().unwrap(), &3)
        assert_eq!(it.next_back().unwrap(), &4)
        assert_eq!(it.next_back().unwrap(), &5)
        assert_eq!(it.next_back().unwrap(), &7)
        assert_eq!(it.next_back(), None)
    }

    #[test]
    fn test_random_access_chain() {
        let xs = [1, 2, 3, 4, 5];
        let ys = ~[7, 9, 11];
        let mut it = xs.iter().chain_(ys.iter());
        assert_eq!(it.idx(0).unwrap(), &1);
        assert_eq!(it.idx(5).unwrap(), &7);
        assert_eq!(it.idx(7).unwrap(), &11);
        assert!(it.idx(8).is_none());

        it.next();
        it.next();
        it.next_back();

        assert_eq!(it.idx(0).unwrap(), &3);
        assert_eq!(it.idx(4).unwrap(), &9);
        assert!(it.idx(6).is_none());
    }
}
