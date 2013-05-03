// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Boolean logic

#[cfg(notest)]
use cmp::{Eq, Ord, TotalOrd, Ordering};
use option::{None, Option, Some};
use from_str::FromStr;

/// Negation / inverse
pub fn not(v: bool) -> bool { !v }

/// Conjunction
pub fn and(a: bool, b: bool) -> bool { a && b }

/// Disjunction
pub fn or(a: bool, b: bool) -> bool { a || b }

/**
 * Exclusive or
 *
 * Identical to `or(and(a, not(b)), and(not(a), b))`
 */
pub fn xor(a: bool, b: bool) -> bool { (a && !b) || (!a && b) }

/// Implication in the logic, i.e. from `a` follows `b`
pub fn implies(a: bool, b: bool) -> bool { !a || b }

/// true if truth values `a` and `b` are indistinguishable in the logic
pub fn eq(a: bool, b: bool) -> bool { a == b }

/// true if truth values `a` and `b` are distinguishable in the logic
pub fn ne(a: bool, b: bool) -> bool { a != b }

/// true if `v` represents truth in the logic
pub fn is_true(v: bool) -> bool { v }

/// true if `v` represents falsehood in the logic
pub fn is_false(v: bool) -> bool { !v }

/// Parse logic value from `s`
impl FromStr for bool {
    fn from_str(s: &str) -> Option<bool> {
        if s == "true" {
            Some(true)
        } else if s == "false" {
            Some(false)
        } else {
            None
        }
    }
}

/// Convert `v` into a string
pub fn to_str(v: bool) -> ~str { if v { ~"true" } else { ~"false" } }

/**
 * Iterates over all truth values by passing them to `blk` in an unspecified
 * order
 */
pub fn all_values(blk: &fn(v: bool)) {
    blk(true);
    blk(false);
}

/// converts truth value to an 8 bit byte
#[inline(always)]
pub fn to_bit(v: bool) -> u8 { if v { 1u8 } else { 0u8 } }

#[cfg(notest)]
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

#[cfg(notest)]
impl TotalOrd for bool {
    #[inline(always)]
    fn cmp(&self, other: &bool) -> Ordering { to_bit(*self).cmp(&to_bit(*other)) }
}

#[cfg(notest)]
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
        assert!(to_str(false) == ~"false");
        assert!(to_str(true) == ~"true");
    }

    #[test]
    fn test_bool_to_bit() {
        do all_values |v| {
            assert!(to_bit(v) == if is_true(v) { 1u8 } else { 0u8 });
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
