// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

/**
* Trait for values that can be compared for equality
* and inequality.
*
* Eventually this may be simplified to only require
* an `eq` method, with the other generated from
* a default implementation. However it should
* remain possible to implement `ne` separately, for
* compatibility with floating-point NaN semantics
* (cf. IEEE 754-2008 section 5.11).
*/
#[lang="eq"]
pub trait Eq {
    pure fn eq(&self, other: &Self) -> bool;
    pure fn ne(&self, other: &Self) -> bool;
}

#[deriving_eq]
pub enum Ordering { Less, Equal, Greater }

/// Trait for types that form a total order
pub trait TotalOrd {
    pure fn cmp(&self, other: &Self) -> Ordering;
}

#[inline(always)]
pure fn icmp<T: Ord>(a: &T, b: &T) -> Ordering {
    if *a < *b { Less }
    else if *a > *b { Greater }
    else { Equal }
}

impl TotalOrd for u8 {
    #[inline(always)]
    pure fn cmp(&self, other: &u8) -> Ordering { icmp(self, other) }
}

impl TotalOrd for u16 {
    #[inline(always)]
    pure fn cmp(&self, other: &u16) -> Ordering { icmp(self, other) }
}

impl TotalOrd for u32 {
    #[inline(always)]
    pure fn cmp(&self, other: &u32) -> Ordering { icmp(self, other) }
}

impl TotalOrd for u64 {
    #[inline(always)]
    pure fn cmp(&self, other: &u64) -> Ordering { icmp(self, other) }
}

impl TotalOrd for i8 {
    #[inline(always)]
    pure fn cmp(&self, other: &i8) -> Ordering { icmp(self, other) }
}

impl TotalOrd for i16 {
    #[inline(always)]
    pure fn cmp(&self, other: &i16) -> Ordering { icmp(self, other) }
}

impl TotalOrd for i32 {
    #[inline(always)]
    pure fn cmp(&self, other: &i32) -> Ordering { icmp(self, other) }
}

impl TotalOrd for i64 {
    #[inline(always)]
    pure fn cmp(&self, other: &i64) -> Ordering { icmp(self, other) }
}

impl TotalOrd for int {
    #[inline(always)]
    pure fn cmp(&self, other: &int) -> Ordering { icmp(self, other) }
}

impl TotalOrd for uint {
    #[inline(always)]
    pure fn cmp(&self, other: &uint) -> Ordering { icmp(self, other) }
}

/**
* Trait for values that can be compared for a sort-order.
*
* Eventually this may be simplified to only require
* an `le` method, with the others generated from
* default implementations. However it should remain
* possible to implement the others separately, for
* compatibility with floating-point NaN semantics
* (cf. IEEE 754-2008 section 5.11).
*/
#[lang="ord"]
pub trait Ord {
    pure fn lt(&self, other: &Self) -> bool;
    pure fn le(&self, other: &Self) -> bool;
    pure fn ge(&self, other: &Self) -> bool;
    pure fn gt(&self, other: &Self) -> bool;
}

#[inline(always)]
pub pure fn lt<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2)
}

#[inline(always)]
pub pure fn le<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).le(v2)
}

#[inline(always)]
pub pure fn eq<T:Eq>(v1: &T, v2: &T) -> bool {
    (*v1).eq(v2)
}

#[inline(always)]
pub pure fn ne<T:Eq>(v1: &T, v2: &T) -> bool {
    (*v1).ne(v2)
}

#[inline(always)]
pub pure fn ge<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).ge(v2)
}

#[inline(always)]
pub pure fn gt<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).gt(v2)
}

/// The equivalence relation. Two values may be equivalent even if they are
/// of different types. The most common use case for this relation is
/// container types; e.g. it is often desirable to be able to use `&str`
/// values to look up entries in a container with `~str` keys.
pub trait Equiv<T> {
    pure fn equiv(&self, other: &T) -> bool;
}

#[inline(always)]
pub pure fn min<T:Ord>(v1: T, v2: T) -> T {
    if v1 < v2 { v1 } else { v2 }
}

#[inline(always)]
pub pure fn max<T:Ord>(v1: T, v2: T) -> T {
    if v1 > v2 { v1 } else { v2 }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_int() {
        fail_unless!(5.cmp(&10) == Less);
        fail_unless!(10.cmp(&5) == Greater);
        fail_unless!(5.cmp(&5) == Equal);
        fail_unless!((-5).cmp(&12) == Less);
        fail_unless!(12.cmp(-5) == Greater);
    }
}
