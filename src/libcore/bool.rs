// -*- rust -*-

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! Boolean logic

import cmp::Eq;

export not, and, or, xor, implies;
export eq, ne, is_true, is_false;
export from_str, to_str, all_values, to_bit;

/// Negation / inverse
pure fn not(v: bool) -> bool { !v }

/// Conjunction
pure fn and(a: bool, b: bool) -> bool { a && b }

/// Disjunction
pure fn or(a: bool, b: bool) -> bool { a || b }

/**
 * Exclusive or
 *
 * Identical to `or(and(a, not(b)), and(not(a), b))`
 */
pure fn xor(a: bool, b: bool) -> bool { (a && !b) || (!a && b) }

/// Implication in the logic, i.e. from `a` follows `b`
pure fn implies(a: bool, b: bool) -> bool { !a || b }

/// true if truth values `a` and `b` are indistinguishable in the logic
pure fn eq(a: bool, b: bool) -> bool { a == b }

/// true if truth values `a` and `b` are distinguishable in the logic
pure fn ne(a: bool, b: bool) -> bool { a != b }

/// true if `v` represents truth in the logic
pure fn is_true(v: bool) -> bool { v }

/// true if `v` represents falsehood in the logic
pure fn is_false(v: bool) -> bool { !v }

/// Parse logic value from `s`
pure fn from_str(s: &str) -> Option<bool> {
    if s == "true" {
        Some(true)
    } else if s == "false" {
        Some(false)
    } else {
        None
    }
}

/// Convert `v` into a string
pure fn to_str(v: bool) -> ~str { if v { ~"true" } else { ~"false" } }

/**
 * Iterates over all truth values by passing them to `blk` in an unspecified
 * order
 */
fn all_values(blk: fn(v: bool)) {
    blk(true);
    blk(false);
}

/// converts truth value to an 8 bit byte
pure fn to_bit(v: bool) -> u8 { if v { 1u8 } else { 0u8 } }

impl bool : cmp::Eq {
    pure fn eq(&&other: bool) -> bool {
        self == other
    }
}

#[test]
fn test_bool_from_str() {
    do all_values |v| {
        assert Some(v) == from_str(bool::to_str(v))
    }
}

#[test]
fn test_bool_to_str() {
    assert to_str(false) == ~"false";
    assert to_str(true) == ~"true";
}

#[test]
fn test_bool_to_bit() {
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
