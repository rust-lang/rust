// Regression test for https://github.com/rust-lang/rust/issues/157401

// ignore-tidy-target-specific-tests
//@ check-fail
//@ compile-flags: --target={{src-base}}/codegen/custom-target-invalid-llvm-target.json -Z unstable-options

fn main() {}

//~? ERROR could not create LLVM MCSubtargetInfo for triple: not-a-real-target
