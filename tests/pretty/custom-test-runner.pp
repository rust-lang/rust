#![feature(prelude_import)]
#![no_std]
//@ compile-flags: --crate-type=lib --test --remap-path-prefix={{src-base}}/=/the/src/ --remap-path-prefix={{src-base}}\=/the/src/
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:custom-test-runner.pp

// Example taken from the unstable book.

#![feature(custom_test_frameworks)]
#![test_runner(my_runner)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

fn my_runner(tests: &[&i32]) {
    for t in tests {
        if **t == 0 {


            { ::std::io::_print(format_args!("PASSED\n")); };
        } else { { ::std::io::_print(format_args!("FAILED\n")); }; }
    }
}
#[rustc_test_marker = "WILL_PASS"]
pub const WILL_PASS: i32 = 0;
#[rustc_test_marker = "WILL_FAIL"]
pub const WILL_FAIL: i32 = 4;
#[rustc_main]
#[coverage(off)]
#[doc(hidden)]
pub fn main() -> () { my_runner(&[&WILL_FAIL, &WILL_PASS]) }
