/*
Module: u32
*/

/*
Const: min_value

Return the minimal value for a u32
*/
const min_value: u32 = 0u32;

/*
Const: max_value

Return the maximal value for a u32
*/
const max_value: u32 = 0xffff_ffffu32;

/* Function: add */
pure fn add(x: u32, y: u32) -> u32 { ret x + y; }

/* Function: sub */
pure fn sub(x: u32, y: u32) -> u32 { ret x - y; }

/* Function: mul */
pure fn mul(x: u32, y: u32) -> u32 { ret x * y; }

/* Function: div */
pure fn div(x: u32, y: u32) -> u32 { ret x / y; }

/* Function: rem */
pure fn rem(x: u32, y: u32) -> u32 { ret x % y; }

/* Predicate: lt */
pure fn lt(x: u32, y: u32) -> bool { ret x < y; }

/* Predicate: le */
pure fn le(x: u32, y: u32) -> bool { ret x <= y; }

/* Predicate: eq */
pure fn eq(x: u32, y: u32) -> bool { ret x == y; }

/* Predicate: ne */
pure fn ne(x: u32, y: u32) -> bool { ret x != y; }

/* Predicate: ge */
pure fn ge(x: u32, y: u32) -> bool { ret x >= y; }

/* Predicate: gt */
pure fn gt(x: u32, y: u32) -> bool { ret x > y; }

/*
Function: range

Iterate over the range [`lo`..`hi`)
*/
fn range(lo: u32, hi: u32, it: block(u32)) {
    let i = lo;
    while i < hi { it(i); i += 1u32; }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
