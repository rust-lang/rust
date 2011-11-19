// -*- rust -*-

/*
Module: four

The fourrternary Belnap relevance logic FOUR represented as ADT

This allows reasoning with four logic values (true, false, none, both)

Implementation: Truth values are represented using a single u8 and
all operations are done using bitshifting which is fast
on current cpus.
*/

import tri;

export t, none, true, false, both;
export not, and, or, xor, implies, implies_materially;
export eq, ne, is_true, is_false;
export from_str, to_str, all_values, to_trit, to_bit;

/*
Type: t

The type of fourrternary logic values

It may be thought of as  tuple `(y, x)` of two bools

*/
type t = u8;

const b0: u8  = 1u8;
const b1: u8  = 2u8;
const b01: u8 = 3u8;

/*
Constant: none

Logic value `(0, 0)` for bottom (neither true or false)
*/
const none: t  = 0u8;

/*
Constant: true

Logic value `(0, 1)` for truth
*/
const true: t  = 1u8;

/*
Constant: false

Logic value `(1, 0)` for falsehood
*/
const false: t = 2u8;

/*
Constant: both

Logic value `(1, 1)` for top (both true and false)
*/
const both: t  = 3u8;

/* Function: not

Negation/Inverse

Returns:

`'(v.y, v.x)`
*/
pure fn not(v: t) -> t { ((v << 1u8) | (v >> 1u8)) & b01 }

/* Function: and

Conjunction

Returns:

`(a.x | b.x, a.y & b.y)`
*/
pure fn and(a: t, b: t) -> t { ((a & b) & b0) | ((a | b) & b1) }

/* Function: or

Disjunction

Returns:

`(a.x & b.x, a.y | b.y)`
*/
pure fn or(a: t, b: t) -> t { ((a | b) & b0) | ((a & b) & b1) }

/* Function: xor

Classic exclusive or

Returns:

`or(and(a, not(b)), and(not(a), b))`
*/
pure fn xor(a: t, b: t) -> t { or(and(a, not(b)), and(not(a), b)) }

/*
Function: implies

Strong implication (from `a` strongly follows `b`)

Returns:

`( x1 & y2, !x1 | x2)`
*/
pure fn implies(a: t, b: t) -> t { ((a << 1u8) & b & b1) | (((!a) | b) & b0) }

/*
Function: implies_materially

Classic (material) implication in the logic
(from `a` materially follows `b`)

Returns:

`or(not(a), b)`
*/
pure fn implies_materially(a: t, b: t) -> t { or(not(a), b) }

/*
Predicate: eq

Returns:

true if truth values `a` and `b` are indistinguishable in the logic
*/
pure fn eq(a: t, b: t) -> bool { a == b }

/*
Predicate: ne

Returns:

true if truth values `a` and `b` are distinguishable in the logic
*/
pure fn ne(a: t, b: t) -> bool { a != b }

/*
Predicate: is_true

Returns:

true if `v` represents truth in the logic (is `true` or `both`)
*/
pure fn is_true(v: t) -> bool { (v & b0) != 0u8 }

/*
Predicate: is_false

Returns:

true if `v` represents falsehood in the logic (is `false` or `none`)
*/
pure fn is_false(v: t) -> bool { (v & b0) == 0u8 }

/*
Function: from_str

Parse logic value from `s`
*/
pure fn from_str(s: str) -> t {
    alt s {
      "none" { none }
      "false" { four::false }
      "true" { four::true }
      "both" { both }
    }
}

/*
Function: to_str

Convert `v` into a string
*/
pure fn to_str(v: t) -> str {
    // FIXME replace with consts as soon as that works
    alt v {
      0u8 { "none" }
      1u8 { "true" }
      2u8 { "false" }
      3u8 { "both" }
    }
}

/*
Function: all_values

Iterates over all truth values by passing them to `blk`
in an unspecified order
*/
fn all_values(blk: block(v: t)) {
    blk(both);
    blk(four::true);
    blk(four::false);
    blk(none);
}

/*
Function: to_bit

Returns:

An u8 whose first bit is set if `if_true(v)` holds
*/
fn to_bit(v: t) -> u8 { v & b0 }

/*
Function: to_tri

Returns:

A trit of `v` (`both` and `none` are both coalesced into `trit::unknown`)
*/
fn to_trit(v: t) -> tri::t { v & (v ^ not(v)) }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
