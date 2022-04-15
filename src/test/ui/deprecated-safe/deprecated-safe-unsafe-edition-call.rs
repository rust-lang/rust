// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{depr_safe_2015, depr_safe_2015_future, depr_safe_2018};

fn main() {
    // NOTE: this test is separate from deprecated-safe-unsafe-edition as the other compiler
    // errors will stop compilation before these calls are checked

    // usage without unsafe should lint
    depr_safe_2015(); //~ ERROR call to unsafe function is unsafe and requires unsafe function or block
    depr_safe_2015_future();
    depr_safe_2018(); //~ WARN use of function `deprecated_safe::depr_safe_2018` without an unsafe function or block has been deprecated as it is now an unsafe function

    // these shouldn't lint, appropriate unsafe usage
    unsafe {
        depr_safe_2015();
    }
    unsafe {
        //~^ WARN unnecessary `unsafe` block
        depr_safe_2015_future();
    }
    unsafe {
        depr_safe_2018();
    }
}
