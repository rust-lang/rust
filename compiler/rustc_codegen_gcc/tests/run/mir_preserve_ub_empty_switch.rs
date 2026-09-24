// ignore-if: test -z "$CARGO_TEST_FLAGS"
// Compiler:
//
// Run-time:
//   status: 0

// Regression test for https://github.com/rust-lang/rustc_codegen_gcc/issues/881
//
// This needs `-Zmir-preserve-ub`, so it is skipped unless that flag is passed
// through `CARGO_TEST_FLAGS` (see the `ignore-if` directive above). Run it with:
//   CARGO_TEST_FLAGS="-Zmir-preserve-ub" ./y.sh test --cargo-tests -- mir_preserve_ub_empty_switch

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use intrinsics::black_box;
use mini_core::*;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    // With `-Zmir-preserve-ub`, the range pattern below is lowered to a pair of
    // comparisons and the second one becomes a `SwitchInt` with no cases (only
    // an `otherwise` target) whose discriminant is the `bool` comparison
    // result. `gcc_jit_block_end_with_switch` rejects a non-integer
    // discriminant, so the backend must emit a plain jump for it instead.
    let value = black_box(argc);
    match value {
        0..=9 => (),
        _ => (),
    }
    0
}
