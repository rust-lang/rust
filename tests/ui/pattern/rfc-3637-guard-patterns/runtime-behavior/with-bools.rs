//! Check for correct runtime behavior when using bools in guard patterns

//@ compile-flags: -Zvalidate-mir -Zlint-mir
//@ run-pass

#![feature(guard_patterns)]
#![allow(incomplete_features)]

fn main() {
    assert!(generic_usage(true, false, true));
    assert!(!generic_usage(false, true, true));
    assert!(with_ops(true, false, true));
    assert!(with_ops(false, true, true))
}

fn generic_usage(x: bool, y: bool, z: bool) -> bool {
    match (x, y) {
        // (true, false, true)
        (true if z, false if z) => true,
        // (false, true, true)
        (false if z, true if z) => false,
        _ => false
    }
}

fn with_ops(x: bool, y: bool, z: bool) -> bool {
    match (x, y) {
        // (true, false, true)
        (true if y || z, false if x && z) => true,
        // (false, true, true)
        (false if y && z, true if y || z) => true,
        _ => false
    }
 }
