//@ edition: 2021
//@ revisions: good
//@ check-pass
//@ compile-flags: -Cinstrument-coverage -Zcoverage-options=mcdc -Zno-profiler-runtime

// Check that we emit some kind of diagnostic when MC/DC instrumentation sees
// code that exceeds the limit of 6 conditions per decision, and falls back
// to only instrumenting that code for branch coverage.
//
// See also `tests/coverage/mcdc/condition-limit.rs`, which tests the actual
// effect on instrumentation.
//
// (The limit is enforced in `compiler/rustc_mir_build/src/build/coverageinfo/mcdc.rs`.)

#[cfg(good)]
fn main() {
    // 7 conditions is allowed, so no diagnostic.
    let [a, b, c, d, e, f, g] = <[bool; 7]>::default();
    if a && b && c && d && e && f && g {
        core::hint::black_box("hello");
    }
}
