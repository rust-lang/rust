//@run-rustfix

#![warn(clippy::filter_next)]
#![allow(clippy::useless_vec)]

/// Checks implementation of `FILTER_NEXT` lint.
fn main() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];

    // Single-line case.
    let _ = v.iter().filter(|&x| *x < 0).next();
}
