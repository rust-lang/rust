// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The `Ord` and `Eq` comparison traits

This module contains the definition of both `Ord` and `Eq` which define
the common interfaces for doing comparison. Both are language items
that the compiler uses to implement the comparison operators. Rust code
may implement `Ord` to overload the `<`, `<=`, `>`, and `>=` operators,
and `Eq` to overload the `==` and `!=` operators.

*/

#[allow(missing_doc)];

/** Equality relation
*
* This trait provides comparisons for equality and inquality.
*
* Requirements:
*
* `a != b` returns the same value as `!(a == b)`
* `a == a` is true
* `a == b` implies `b == a`
* `a == b && b == c` implies `a == c`
*
* Eq only requires the `eq` method to be implemented; `ne` is its negation by default.
*/
#[lang="eq"]
pub trait Eq {
    fn eq(&self, other: &Self) -> bool;

    #[inline]
    fn ne(&self, other: &Self) -> bool { !self.eq(other) }
}

#[deriving(Clone, Eq)]
pub enum Ordering { Less = -1, Equal = 0, Greater = 1 }

impl Ord for Ordering {
    #[inline]
    fn lt(&self, other: &Ordering) -> bool { (*self as int) < (*other as int) }
}

/// Compares (a1, b1) against (a2, b2), where the a values are more significant.
pub fn cmp2<A: Ord, B: Ord>(
    a1: &A, b1: &B,
    a2: &A, b2: &B) -> Ordering
{
    match a1.cmp(a2) {
        Less => Less,
        Greater => Greater,
        Equal => b1.cmp(b2)
    }
}

/**
Return `o1` if it is not `Equal`, otherwise `o2`. Simulates the
lexical ordering on a type `(int, int)`.
*/
#[inline]
pub fn lexical_ordering(o1: Ordering, o2: Ordering) -> Ordering {
    match o1 {
        Equal => o2,
        _ => o1
    }
}

/** Strict total ordering
*
* This trait provides sort order comparisons.
*
* Ord only requires implementation of the `lt` method, with the others
* generated from default implementations.
*
* The `cmp` method allows for an efficient three-way comparison implementation.
*
* The default implementation of `cmp` should be overridden with a single-pass
* implementation for ordered container types (vectors, strings, tree-based
* maps/sets).
*
* Generic wrapper types (like smart pointers) should override `cmp` and call
* the underlying `cmp` method for efficiency. A derived implementation will do
* this automatically.
*/
#[lang="ord"]
pub trait Ord: Eq {
    fn lt(&self, other: &Self) -> bool;

    #[inline]
    fn le(&self, other: &Self) -> bool { !other.lt(self) }

    #[inline]
    fn gt(&self, other: &Self) -> bool {  other.lt(self) }

    #[inline]
    fn ge(&self, other: &Self) -> bool { !self.lt(other) }

    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if *self < *other { Less }
        else if *self > *other { Greater }
        else { Equal }
    }

    // FIXME (#12068): Add min/max/clamp default methods
}

/// The equivalence relation. Two values may be equivalent even if they are
/// of different types. The most common use case for this relation is
/// container types; e.g. it is often desirable to be able to use `&str`
/// values to look up entries in a container with `~str` keys.
pub trait Equiv<T> {
    fn equiv(&self, other: &T) -> bool;
}

#[inline]
pub fn min<T:Ord>(v1: T, v2: T) -> T {
    if v1 < v2 { v1 } else { v2 }
}

#[inline]
pub fn max<T:Ord>(v1: T, v2: T) -> T {
    if v1 > v2 { v1 } else { v2 }
}

#[cfg(test)]
mod test {
    use super::lexical_ordering;

    #[test]
    fn test_int_cmp() {
        assert_eq!(5.cmp(&10), Less);
        assert_eq!(10.cmp(&5), Greater);
        assert_eq!(5.cmp(&5), Equal);
        assert_eq!((-5).cmp(&12), Less);
        assert_eq!(12.cmp(-5), Greater);
    }

    #[test]
    fn test_cmp2() {
        assert_eq!(cmp2(1, 2, 3, 4), Less);
        assert_eq!(cmp2(3, 2, 3, 4), Less);
        assert_eq!(cmp2(5, 2, 3, 4), Greater);
        assert_eq!(cmp2(5, 5, 5, 4), Greater);
    }

    #[test]
    fn test_int_totaleq() {
        assert!(5.equals(&5));
        assert!(!2.equals(&17));
    }

    #[test]
    fn test_ordering_order() {
        assert!(Less < Equal);
        assert_eq!(Greater.cmp(&Less), Greater);
    }

    #[test]
    fn test_lexical_ordering() {
        fn t(o1: Ordering, o2: Ordering, e: Ordering) {
            assert_eq!(lexical_ordering(o1, o2), e);
        }

        let xs = [Less, Equal, Greater];
        for &o in xs.iter() {
            t(Less, o, Less);
            t(Equal, o, o);
            t(Greater, o, Greater);
         }
    }
}
