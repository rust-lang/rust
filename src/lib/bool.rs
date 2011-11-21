// -*- rust -*-

/*
Module: bool

Classic Boolean logic reified as ADT
*/

export t;
export not, and, or, xor, implies;
export eq, ne, is_true, is_false;
export from_str, to_str, all_values, to_bit;

/*
Type: t

The type of boolean logic values
*/
type t = bool;

/* Function: not

Negation/Inverse
*/
pure fn not(v: t) -> t { !v }

/* Function: and

Conjunction
*/
pure fn and(a: t, b: t) -> t { a && b }

/* Function: and

Disjunction
*/
pure fn or(a: t, b: t) -> t { a || b }

/*
Function: xor

Exclusive or, i.e. `or(and(a, not(b)), and(not(a), b))`
*/
pure fn xor(a: t, b: t) -> t { (a && !b) || (!a && b) }

/*
Function: implies

Implication in the logic, i.e. from `a` follows `b`
*/
pure fn implies(a: t, b: t) -> t { !a || b }

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

true if `v` represents truth in the logic
*/
pure fn is_true(v: t) -> bool { v }

/*
Predicate: is_false

Returns:

true if `v` represents falsehood in the logic
*/
pure fn is_false(v: t) -> bool { !v }

/*
Function: from_str

Parse logic value from `s`
*/
pure fn from_str(s: str) -> t {
    alt s {
      "true" { true }
      "false" { false }
    }
}

/*
Function: to_str

Convert `v` into a string
*/
pure fn to_str(v: t) -> str { if v { "true" } else { "false" } }

/*
Function: all_values

Iterates over all truth values by passing them to `blk`
in an unspecified order
*/
fn all_values(blk: block(v: t)) {
    blk(true);
    blk(false);
}

/*
Function: to_bit

Returns:

An u8 whose first bit is set if `if_true(v)` holds
*/
fn to_bit(v: t) -> u8 { if v { 1u8 } else { 0u8 } }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
