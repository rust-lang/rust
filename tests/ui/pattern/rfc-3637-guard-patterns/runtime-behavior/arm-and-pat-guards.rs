//! Check for correct runtime behavior when using guard patterns combined with arm guards

//@ compile-flags: -Zvalidate-mir -Zlint-mir
//@ run-pass

#![feature(guard_patterns)]
#![allow(incomplete_features)]

fn main() {
    assert!(guard_arm_pats(true, false, true));
    assert!(guard_arm_pats(false, true, true));
}

fn guard_arm_pats(x: bool, y: bool, z: bool) -> bool {
    match (x, y) {
        // (true, false, true)
        (true if x, false if x) if z => true,
        // (false, true, true)
        (false if z, true if !x) if y => true,
        _ => false
    }
}
