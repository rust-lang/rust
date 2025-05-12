//@ run-pass
#![allow(non_camel_case_types)]

// Test that unboxed closures work with cross-crate inlining
// Acts as a regression test for #16790, #18378 and #18543

//@ aux-build:unboxed-closures-cross-crate.rs

extern crate unboxed_closures_cross_crate as ubcc;

fn main() {
    assert_eq!(ubcc::has_closures(), 2_usize);
    assert_eq!(ubcc::has_generic_closures(2_usize, 3_usize), 5_usize);
}
