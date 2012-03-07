#[doc = "Operations and constants for `u8`"];

/*
Module: u8
*/

/*
Const: min_value

The minumum value of a u8.
*/
const min_value: u8 = 0u8;

/*
Const: max_value

The maximum value of a u8.
*/
const max_value: u8 = 0u8 - 1u8;

/* Function: add */
pure fn add(x: u8, y: u8) -> u8 { ret x + y; }

/* Function: sub */
pure fn sub(x: u8, y: u8) -> u8 { ret x - y; }

/* Function: mul */
pure fn mul(x: u8, y: u8) -> u8 { ret x * y; }

/* Function: div */
pure fn div(x: u8, y: u8) -> u8 { ret x / y; }

/* Function: rem */
pure fn rem(x: u8, y: u8) -> u8 { ret x % y; }

/* Predicate: lt */
pure fn lt(x: u8, y: u8) -> bool { ret x < y; }

/* Predicate: le */
pure fn le(x: u8, y: u8) -> bool { ret x <= y; }

/* Predicate: eq */
pure fn eq(x: u8, y: u8) -> bool { ret x == y; }

/* Predicate: ne */
pure fn ne(x: u8, y: u8) -> bool { ret x != y; }

/* Predicate: ge */
pure fn ge(x: u8, y: u8) -> bool { ret x >= y; }

/* Predicate: gt */
pure fn gt(x: u8, y: u8) -> bool { ret x > y; }

/* Predicate: is_ascii */
pure fn is_ascii(x: u8) -> bool { ret 0u8 == x & 128u8; }

/*
Function: range

Iterate over the range [`lo`..`hi`)
*/
fn range(lo: u8, hi: u8, it: fn(u8)) {
    let mut i = lo;
    while i < hi { it(i); i += 1u8; }
}

#[doc = "Computes the bitwise complement"]
fn compl(i: u8) -> u8 {
    max_value ^ i
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
