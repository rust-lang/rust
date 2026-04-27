//! Check for correct runtime behavior when using guard patterns with or-patterns

//@ compile-flags: -Zvalidate-mir -Zlint-mir
//@ run-pass

#![feature(guard_patterns)]
#![allow(incomplete_features)]

fn main() {
    assert!(arr_with_or_pats([1, 2, 4], true, true, false, true, true));
    assert!(arr_with_or_pats([1, 3, 4], true, false, true, true, true));
}

fn arr_with_or_pats(arr: [u8; 3], a: bool, b: bool, c: bool, d: bool, e: bool) -> bool {
    const A: u8 = 1;
    const B: u8 = 2;
    const C: u8 = 3;
    const D: u8 = 4;

    match arr {
        // [1, 2 | 3, 4]
        [A if a, (B if b) | (C if c), D if d] if e => true,
        _ => false
    }
}
