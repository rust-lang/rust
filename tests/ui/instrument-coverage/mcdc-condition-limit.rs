//@ edition: 2021
//@ revisions: good bad
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
    // 6 conditions is OK, so no diagnostic.
    let [a, b, c, d, e, f] = <[bool; 6]>::default();
    if a && b && c && d && e && f {
        core::hint::black_box("hello");
    }
}

#[cfg(bad)]
fn main() {
    // 7 conditions is too many, so issue a diagnostic.
    let [a, b, c, d, e, f, g] = <[bool; 7]>::default();
    if a && b && c && d && e && f && g { //[bad]~ WARNING number of conditions in decision
        core::hint::black_box("hello");
    }
}
