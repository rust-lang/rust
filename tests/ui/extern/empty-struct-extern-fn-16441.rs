//! Regression test for https://github.com/rust-lang/rust/issues/16441

//@ run-pass
#![allow(dead_code)]

struct Empty;

// This used to cause an ICE
#[allow(improper_ctypes_definitions)]
extern "C" fn ice(_a: Empty) {}

fn main() {
}
