// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! Boolean logic

use bool;
use cmp;
use cmp::Eq;
use option::{None, Option, Some};

/// Negation / inverse
pub pure fn not(v: bool) -> bool { !v }

/// Conjunction
pub pure fn and(a: bool, b: bool) -> bool { a && b }

/// Disjunction
pub pure fn or(a: bool, b: bool) -> bool { a || b }

/**
 * Exclusive or
 *
 * Identical to `or(and(a, not(b)), and(not(a), b))`
 */
pub pure fn xor(a: bool, b: bool) -> bool { (a && !b) || (!a && b) }

/// Implication in the logic, i.e. from `a` follows `b`
pub pure fn implies(a: bool, b: bool) -> bool { !a || b }

/// true if truth values `a` and `b` are indistinguishable in the logic
pub pure fn eq(a: bool, b: bool) -> bool { a == b }

/// true if truth values `a` and `b` are distinguishable in the logic
pub pure fn ne(a: bool, b: bool) -> bool { a != b }

/// true if `v` represents truth in the logic
pub pure fn is_true(v: bool) -> bool { v }

/// true if `v` represents falsehood in the logic
pub pure fn is_false(v: bool) -> bool { !v }

/// Parse logic value from `s`
pub pure fn from_str(s: &str) -> Option<bool> {
    if s == "true" {
        Some(true)
    } else if s == "false" {
        Some(false)
    } else {
        None
    }
}

/// Convert `v` into a string
pub pure fn to_str(v: bool) -> ~str { if v { ~"true" } else { ~"false" } }

/**
 * Iterates over all truth values by passing them to `blk` in an unspecified
 * order
 */
pub fn all_values(blk: fn(v: bool)) {
    blk(true);
    blk(false);
}

/// converts truth value to an 8 bit byte
pub pure fn to_bit(v: bool) -> u8 { if v { 1u8 } else { 0u8 } }

#[cfg(notest)]
impl bool : cmp::Eq {
    pure fn eq(&self, other: &bool) -> bool { (*self) == (*other) }
    pure fn ne(&self, other: &bool) -> bool { (*self) != (*other) }
}

#[test]
pub fn test_bool_from_str() {
    do all_values |v| {
        assert Some(v) == from_str(bool::to_str(v))
    }
}

#[test]
pub fn test_bool_to_str() {
    assert to_str(false) == ~"false";
    assert to_str(true) == ~"true";
}

#[test]
pub fn test_bool_to_bit() {
    do all_values |v| {
        assert to_bit(v) == if is_true(v) { 1u8 } else { 0u8 };
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
