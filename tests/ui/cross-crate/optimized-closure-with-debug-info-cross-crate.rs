//@ run-pass
//@ aux-build:optimized-closure-with-debug-info-cross-crate-1.rs
//@ aux-build:optimized-closure-with-debug-info-cross-crate-2.rs

//! Regression test for https://github.com/rust-lang/rust/issues/31702
// this test is actually entirely in the linked library crates

extern crate optimized_closure_with_debug_info_cross_crate_1;
extern crate optimized_closure_with_debug_info_cross_crate_2;

fn main() {}
