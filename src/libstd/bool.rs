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

The `bool` module contains useful code to help work with boolean values.

A quick summary:

## Trait implementations for `bool`

Implementations of the following traits:

* `FromStr`
* `Ord`
* `TotalOrd`
* `Eq`

## Various functions to compare `bool`s

All of the standard comparison functions one would expect: `and`, `eq`, `or`,
and more.

Also, a few conversion functions: `to_bit` and `to_str`.

Finally, some inquries into the nature of truth: `is_true` and `is_false`.

*/

#[cfg(not(test))]
use cmp::{Eq, Ord, TotalOrd, Ordering};
use option::{None, Option, Some};
use from_str::FromStr;

/**
* Negation of a boolean value.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::not(true)
* false
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::not(false)
* true
* ~~~
*/
pub fn not(v: bool) -> bool { !v }

/**
* Conjunction of two boolean values.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::and(true, false)
* false
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::and(true, true)
* true
* ~~~
*/
pub fn and(a: bool, b: bool) -> bool { a && b }

/**
* Disjunction of two boolean values.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::or(true, false)
* true
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::or(false, false)
* false
* ~~~
*/
pub fn or(a: bool, b: bool) -> bool { a || b }

/**
* An 'exclusive or' of two boolean values.
*
* 'exclusive or' is identical to `or(and(a, not(b)), and(not(a), b))`.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::xor(true, false)
* true
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::xor(true, true)
* false
* ~~~
*/
pub fn xor(a: bool, b: bool) -> bool { (a && !b) || (!a && b) }

/**
* Implication between two boolean values.
*
* Implication is often phrased as 'if a then b.'
*
* 'if a then b' is equivalent to `!a || b`.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::implies(true, true)
* true
*
* ~~~ {.rust}
* rusti> std::bool::implies(true, false)
* false
* ~~~
*/
pub fn implies(a: bool, b: bool) -> bool { !a || b }

/**
* Equality between two boolean values.
*
* Two booleans are equal if they have the same value.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::eq(false, true)
* false
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::eq(false, false)
* true
* ~~~
*/
pub fn eq(a: bool, b: bool) -> bool { a == b }

/**
* Non-equality between two boolean values.
*
* Two booleans are not equal if they have different values.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::ne(false, true)
* true
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::ne(false, false)
* false
* ~~~
*/
pub fn ne(a: bool, b: bool) -> bool { a != b }

/**
* Is a given boolean value true?
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::is_true(true)
* true
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::is_true(false)
* false
* ~~~
*/
pub fn is_true(v: bool) -> bool { v }

/**
* Is a given boolean value false?
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::is_false(false)
* true
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::is_false(true)
* false
* ~~~
*/
pub fn is_false(v: bool) -> bool { !v }

/**
* Parse a `bool` from a `str`.
*
* Yields an `Option<bool>`, because `str` may or may not actually be parseable.
*
* # Examples
*
* ~~~ {.rust}
* rusti> FromStr::from_str::<bool>("true")
* Some(true)
* ~~~
*
* ~~~ {.rust}
* rusti> FromStr::from_str::<bool>("false")
* Some(false)
* ~~~
*
* ~~~ {.rust}
* rusti> FromStr::from_str::<bool>("not even a boolean")
* None
* ~~~
*/
impl FromStr for bool {
    fn from_str(s: &str) -> Option<bool> {
        match s {
            "true"  => Some(true),
            "false" => Some(false),
            _       => None,
        }
    }
}

/**
* Convert a `bool` to a `str`.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::to_str(true)
* "true"
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::to_str(false)
* "false"
* ~~~
*/
pub fn to_str(v: bool) -> ~str { if v { ~"true" } else { ~"false" } }

/**
* Iterates over all truth values, passing them to the given block.
*
* There are no guarantees about the order values will be given.
*
* # Examples
* ~~~
* do std::bool::all_values |x: bool| {
*     println(std::bool::to_str(x));
* }
* ~~~
*/
pub fn all_values(blk: &fn(v: bool)) {
    blk(true);
    blk(false);
}

/**
* Convert a `bool` to a `u8`.
*
* # Examples
*
* ~~~ {.rust}
* rusti> std::bool::to_bit(true)
* 1
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::to_bit(false)
* 0
* ~~~
*/
#[inline(always)]
pub fn to_bit(v: bool) -> u8 { if v { 1u8 } else { 0u8 } }

#[cfg(not(test))]
impl Ord for bool {
    #[inline(always)]
    fn lt(&self, other: &bool) -> bool { to_bit(*self) < to_bit(*other) }
    #[inline(always)]
    fn le(&self, other: &bool) -> bool { to_bit(*self) <= to_bit(*other) }
    #[inline(always)]
    fn gt(&self, other: &bool) -> bool { to_bit(*self) > to_bit(*other) }
    #[inline(always)]
    fn ge(&self, other: &bool) -> bool { to_bit(*self) >= to_bit(*other) }
}

#[cfg(not(test))]
impl TotalOrd for bool {
    #[inline(always)]
    fn cmp(&self, other: &bool) -> Ordering { to_bit(*self).cmp(&to_bit(*other)) }
}

#[cfg(not(test))]
impl Eq for bool {
    #[inline(always)]
    fn eq(&self, other: &bool) -> bool { (*self) == (*other) }
    #[inline(always)]
    fn ne(&self, other: &bool) -> bool { (*self) != (*other) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    #[test]
    fn test_bool_from_str() {
        do all_values |v| {
            assert!(Some(v) == FromStr::from_str(to_str(v)))
        }
    }

    #[test]
    fn test_bool_to_str() {
        assert_eq!(to_str(false), ~"false");
        assert_eq!(to_str(true), ~"true");
    }

    #[test]
    fn test_bool_to_bit() {
        do all_values |v| {
            assert_eq!(to_bit(v), if is_true(v) { 1u8 } else { 0u8 });
        }
    }

    #[test]
    fn test_bool_ord() {
        assert!(true > false);
        assert!(!(false > true));

        assert!(false < true);
        assert!(!(true < false));

        assert!(false <= false);
        assert!(false >= false);
        assert!(true <= true);
        assert!(true >= true);

        assert!(false <= true);
        assert!(!(false >= true));
        assert!(true >= false);
        assert!(!(true <= false));
    }

    #[test]
    fn test_bool_totalord() {
        assert_eq!(true.cmp(&true), Equal);
        assert_eq!(false.cmp(&false), Equal);
        assert_eq!(true.cmp(&false), Greater);
        assert_eq!(false.cmp(&true), Less);
    }
}
