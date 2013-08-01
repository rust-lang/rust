// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Composable internal iterators

Internal iterators are functions implementing the protocol used by the `for` loop.

An internal iterator takes `fn(...) -> bool` as a parameter, with returning `false` used to signal
breaking out of iteration. The adaptors in the module work with any such iterator, not just ones
tied to specific traits. For example:

~~~ {.rust}
println(iter::to_vec(|f| uint::range(0, 20, f)).to_str());
~~~

An external iterator object implementing the interface in the `iterator` module can be used as an
internal iterator by calling the `advance` method. For example:

~~~ {.rust}
let xs = [0u, 1, 2, 3, 4, 5];
let ys = [30, 40, 50, 60];
let mut it = xs.iter().chain(ys.iter());
foreach &x: &uint in it {
    println(x.to_str());
}
~~~

Internal iterators provide a subset of the functionality of an external iterator. It's not possible
to interleave them to implement algorithms like `zip`, `union` and `merge`. However, they're often
much easier to implement.

*/

use std::vec;
use std::cmp::Ord;
use std::option::{Option, Some, None};
use std::num::{One, Zero};
use std::ops::{Add, Mul};

#[allow(missing_doc)]
pub trait FromIter<T> {
    /// Build a container with elements from an internal iterator.
    ///
    /// # Example:
    ///
    /// ~~~ {.rust}
    /// let xs = ~[1, 2, 3];
    /// let ys: ~[int] = do FromIter::from_iter |f| { xs.iter().advance(|x| f(*x)) };
    /// assert_eq!(xs, ys);
    /// ~~~
    pub fn from_iter(iter: &fn(f: &fn(T) -> bool) -> bool) -> Self;
}

/**
 * Return true if `predicate` is true for any values yielded by an internal iterator.
 *
 * Example:
 *
 * ~~~ {.rust}
 * let xs = ~[1u, 2, 3, 4, 5];
 * assert!(any(|&x: &uint| x > 2, |f| xs.iter().advance(f)));
 * assert!(!any(|&x: &uint| x > 5, |f| xs.iter().advance(f)));
 * ~~~
 */
#[inline]
pub fn any<T>(predicate: &fn(T) -> bool,
              iter: &fn(f: &fn(T) -> bool) -> bool) -> bool {
    for iter |x| {
        if predicate(x) {
            return true;
        }
    }
    return false;
}

/**
 * Return true if `predicate` is true for all values yielded by an internal iterator.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * assert!(all(|&x: &uint| x < 6, |f| uint::range(1, 6, f)));
 * assert!(!all(|&x: &uint| x < 5, |f| uint::range(1, 6, f)));
 * ~~~
 */
#[inline]
pub fn all<T>(predicate: &fn(T) -> bool,
              iter: &fn(f: &fn(T) -> bool) -> bool) -> bool {
    // If we ever break, iter will return false, so this will only return true
    // if predicate returns true for everything.
    iter(|x| predicate(x))
}

/**
 * Return the first element where `predicate` returns `true`. Return `None` if no element is found.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * let xs = ~[1u, 2, 3, 4, 5, 6];
 * assert_eq!(*find(|& &x: & &uint| x > 3, |f| xs.iter().advance(f)).unwrap(), 4);
 * ~~~
 */
#[inline]
pub fn find<T>(predicate: &fn(&T) -> bool,
               iter: &fn(f: &fn(T) -> bool) -> bool) -> Option<T> {
    for iter |x| {
        if predicate(&x) {
            return Some(x);
        }
    }
    None
}

/**
 * Return the largest item yielded by an iterator. Return `None` if the iterator is empty.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * let xs = ~[8, 2, 3, 1, -5, 9, 11, 15];
 * assert_eq!(max(|f| xs.iter().advance(f)).unwrap(), &15);
 * ~~~
 */
#[inline]
pub fn max<T: Ord>(iter: &fn(f: &fn(T) -> bool) -> bool) -> Option<T> {
    let mut result = None;
    for iter |x| {
        match result {
            Some(ref mut y) => {
                if x > *y {
                    *y = x;
                }
            }
            None => result = Some(x)
        }
    }
    result
}

/**
 * Return the smallest item yielded by an iterator. Return `None` if the iterator is empty.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * let xs = ~[8, 2, 3, 1, -5, 9, 11, 15];
 * assert_eq!(max(|f| xs.iter().advance(f)).unwrap(), &-5);
 * ~~~
 */
#[inline]
pub fn min<T: Ord>(iter: &fn(f: &fn(T) -> bool) -> bool) -> Option<T> {
    let mut result = None;
    for iter |x| {
        match result {
            Some(ref mut y) => {
                if x < *y {
                    *y = x;
                }
            }
            None => result = Some(x)
        }
    }
    result
}

/**
 * Reduce an iterator to an accumulated value.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * assert_eq!(fold(0i, |f| int::range(1, 5, f), |a, x| *a += x), 10);
 * ~~~
 */
#[inline]
pub fn fold<T, U>(start: T, iter: &fn(f: &fn(U) -> bool) -> bool, f: &fn(&mut T, U)) -> T {
    let mut result = start;
    for iter |x| {
        f(&mut result, x);
    }
    result
}

/**
 * Reduce an iterator to an accumulated value.
 *
 * `fold_ref` is usable in some generic functions where `fold` is too lenient to type-check, but it
 * forces the iterator to yield borrowed pointers.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * fn product<T: One + Mul<T, T>>(iter: &fn(f: &fn(&T) -> bool) -> bool) -> T {
 *     fold_ref(One::one::<T>(), iter, |a, x| *a = a.mul(x))
 * }
 * ~~~
 */
#[inline]
pub fn fold_ref<T, U>(start: T, iter: &fn(f: &fn(&U) -> bool) -> bool, f: &fn(&mut T, &U)) -> T {
    let mut result = start;
    for iter |x| {
        f(&mut result, x);
    }
    result
}

/**
 * Return the sum of the items yielding by an iterator.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * let xs: ~[int] = ~[1, 2, 3, 4];
 * assert_eq!(do sum |f| { xs.iter().advance(f) }, 10);
 * ~~~
 */
#[inline]
pub fn sum<T: Zero + Add<T, T>>(iter: &fn(f: &fn(&T) -> bool) -> bool) -> T {
    fold_ref(Zero::zero::<T>(), iter, |a, x| *a = a.add(x))
}

/**
 * Return the product of the items yielded by an iterator.
 *
 * # Example:
 *
 * ~~~ {.rust}
 * let xs: ~[int] = ~[1, 2, 3, 4];
 * assert_eq!(do product |f| { xs.iter().advance(f) }, 24);
 * ~~~
 */
#[inline]
pub fn product<T: One + Mul<T, T>>(iter: &fn(f: &fn(&T) -> bool) -> bool) -> T {
    fold_ref(One::one::<T>(), iter, |a, x| *a = a.mul(x))
}

impl<T> FromIter<T> for ~[T]{
    #[inline]
    pub fn from_iter(iter: &fn(f: &fn(T) -> bool) -> bool) -> ~[T] {
        let mut v = ~[];
        for iter |x| { v.push(x) }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    use int;
    use uint;

    #[test]
    fn test_from_iter() {
        let xs = ~[1, 2, 3];
        let ys: ~[int] = do FromIter::from_iter |f| { xs.iter().advance(|x| f(*x)) };
        assert_eq!(xs, ys);
    }

    #[test]
    fn test_any() {
        let xs = ~[1u, 2, 3, 4, 5];
        assert!(any(|&x: &uint| x > 2, |f| xs.iter().advance(f)));
        assert!(!any(|&x: &uint| x > 5, |f| xs.iter().advance(f)));
    }

    #[test]
    fn test_all() {
        assert!(all(|x: uint| x < 6, |f| uint::range(1, 6, f)));
        assert!(!all(|x: uint| x < 5, |f| uint::range(1, 6, f)));
    }

    #[test]
    fn test_find() {
        let xs = ~[1u, 2, 3, 4, 5, 6];
        assert_eq!(*find(|& &x: & &uint| x > 3, |f| xs.iter().advance(f)).unwrap(), 4);
    }

    #[test]
    fn test_max() {
        let xs = ~[8, 2, 3, 1, -5, 9, 11, 15];
        assert_eq!(max(|f| xs.iter().advance(f)).unwrap(), &15);
    }

    #[test]
    fn test_min() {
        let xs = ~[8, 2, 3, 1, -5, 9, 11, 15];
        assert_eq!(min(|f| xs.iter().advance(f)).unwrap(), &-5);
    }

    #[test]
    fn test_fold() {
        assert_eq!(fold(0i, |f| int::range(1, 5, f), |a, x| *a += x), 10);
    }

    #[test]
    fn test_sum() {
        let xs: ~[int] = ~[1, 2, 3, 4];
        assert_eq!(do sum |f| { xs.iter().advance(f) }, 10);
    }

    #[test]
    fn test_empty_sum() {
        let xs: ~[int] = ~[];
        assert_eq!(do sum |f| { xs.iter().advance(f) }, 0);
    }

    #[test]
    fn test_product() {
        let xs: ~[int] = ~[1, 2, 3, 4];
        assert_eq!(do product |f| { xs.iter().advance(f) }, 24);
    }

    #[test]
    fn test_empty_product() {
        let xs: ~[int] = ~[];
        assert_eq!(do product |f| { xs.iter().advance(f) }, 1);
    }
}
