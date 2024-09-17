#![feature(coverage_attribute)]
//@ edition: 2021
//@ ignore-llvm-version: 19 - 99
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

// Check that MC/DC instrumentation can gracefully handle conditions that
// exceed LLVM's limit of 6 conditions per decision.
//
// (The limit is enforced in `compiler/rustc_mir_build/src/build/coverageinfo/mcdc.rs`.)

fn good() {
    // With only 6 conditions, perform full MC/DC instrumentation.
    let [a, b, c, d, e, f] = <[bool; 6]>::default();
    if a && b && c && d && e && f {
        core::hint::black_box("hello");
    }
}

fn bad() {
    // With 7 conditions, fall back to branch instrumentation only.
    let [a, b, c, d, e, f, g] = <[bool; 7]>::default();
    if a && b && c && d && e && f && g {
        core::hint::black_box("hello");
    }
}

#[coverage(off)]
fn main() {
    good();
    bad();
}
