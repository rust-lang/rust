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
* `ToStr`
* `Not`
* `Ord`
* `TotalOrd`
* `Eq`
* `Zero`

## Various functions to compare `bool`s

All of the standard comparison functions one would expect: `and`, `eq`, `or`,
and more.

Also, a few conversion functions: `to_bit` and `to_str`.

Finally, some inquries into the nature of truth: `is_true` and `is_false`.

*/

use option::{None, Option, Some};
use from_str::FromStr;
use to_str::ToStr;

#[cfg(not(test))] use cmp::{Eq, Ord, TotalOrd, Ordering};
#[cfg(not(test))] use ops::Not;
#[cfg(not(test))] use num::Zero;

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
* ~~~
*
* ~~~ {.rust}
* rusti> std::bool::implies(true, false)
* false
* ~~~
*/
pub fn implies(a: bool, b: bool) -> bool { !a || b }

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
* rusti> true.to_str()
* "true"
* ~~~
*
* ~~~ {.rust}
* rusti> false.to_str()
* "false"
* ~~~
*/
impl ToStr for bool {
    #[inline]
    fn to_str(&self) -> ~str {
        if *self { ~"true" } else { ~"false" }
    }
}

/**
* Iterates over all truth values, passing them to the given block.
*
* There are no guarantees about the order values will be given.
*
* # Examples
* ~~~
* do std::bool::all_values |x: bool| {
*     println(x.to_str())
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
#[inline]
pub fn to_bit(v: bool) -> u8 { if v { 1u8 } else { 0u8 } }

/**
* The logical complement of a boolean value.
*
* # Examples
*
* ~~~rust
* rusti> !true
* false
* ~~~
*
* ~~~rust
* rusti> !false
* true
* ~~~
*/
#[cfg(not(test))]
impl Not<bool> for bool {
    #[inline]
    fn not(&self) -> bool { !*self }
}

#[cfg(not(test))]
impl Ord for bool {
    #[inline]
    fn lt(&self, other: &bool) -> bool { to_bit(*self) < to_bit(*other) }
    #[inline]
    fn le(&self, other: &bool) -> bool { to_bit(*self) <= to_bit(*other) }
    #[inline]
    fn gt(&self, other: &bool) -> bool { to_bit(*self) > to_bit(*other) }
    #[inline]
    fn ge(&self, other: &bool) -> bool { to_bit(*self) >= to_bit(*other) }
}

#[cfg(not(test))]
impl TotalOrd for bool {
    #[inline]
    fn cmp(&self, other: &bool) -> Ordering { to_bit(*self).cmp(&to_bit(*other)) }
}

/**
* Equality between two boolean values.
*
* Two booleans are equal if they have the same value.
*
* ~~~ {.rust}
* rusti> false.eq(&true)
* false
* ~~~
*
* ~~~ {.rust}
* rusti> false == false
* true
* ~~~
*
* ~~~ {.rust}
* rusti> false != true
* true
* ~~~
*
* ~~~ {.rust}
* rusti> false.ne(&false)
* false
* ~~~
*/
#[cfg(not(test))]
impl Eq for bool {
    #[inline]
    fn eq(&self, other: &bool) -> bool { (*self) == (*other) }
    #[inline]
    fn ne(&self, other: &bool) -> bool { (*self) != (*other) }
}

#[cfg(not(test))]
impl Zero for bool {
    fn zero() -> bool { false }
    fn is_zero(&self) -> bool { *self == false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;

    #[test]
    fn test_bool_from_str() {
        do all_values |v| {
            assert!(Some(v) == FromStr::from_str(v.to_str()))
        }
    }

    #[test]
    fn test_bool_to_str() {
        assert_eq!(false.to_str(), ~"false");
        assert_eq!(true.to_str(), ~"true");
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
