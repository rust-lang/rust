//! Check MIR correctness for general guard patterns usage

//@ compile-flags: -Zvalidate-mir -Zlint-mir
//@ run-pass

#![feature(guard_patterns)]
#![allow(incomplete_features)]

fn main() {
    generic_usage(true, false, true);
}

fn generic_usage(x: bool, y: bool, z: bool) -> bool {
    match (x, y) {
        (true if z, false if !z) => true,
        (false if z, true if z) => false,
        (true, true) => true,
        (false, false) => false
    }
}
