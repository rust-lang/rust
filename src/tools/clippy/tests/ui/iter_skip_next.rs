// run-rustfix
// aux-build:option_helpers.rs

#![warn(clippy::iter_skip_next)]
#![allow(clippy::blacklisted_name)]
#![allow(clippy::iter_nth)]

extern crate option_helpers;

use option_helpers::IteratorFalsePositives;

/// Checks implementation of `ITER_SKIP_NEXT` lint
fn main() {
    let some_vec = vec![0, 1, 2, 3];
    let _ = some_vec.iter().skip(42).next();
    let _ = some_vec.iter().cycle().skip(42).next();
    let _ = (1..10).skip(10).next();
    let _ = &some_vec[..].iter().skip(3).next();
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.skip(42).next();
    let _ = foo.filter().skip(42).next();
}
