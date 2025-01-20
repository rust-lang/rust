//@ edition: 2021
// skip-filecheck

// Under the Rust 2021 disjoint capture rules, a "captured" place sometimes
// doesn't actually need to be captured, if it is only matched against
// irrefutable patterns that don't bind anything.
//
// When that happens, there is currently some MIR-building code
// (`Builder::prefix_slice_suffix`) that can no longer distinguish between
// array patterns and slice patterns, so it falls back to the code for dealing
// with slice patterns.
//
// That appears to be benign, but it's worth having a test that explicitly
// triggers the edge-case scenario. If someone makes a change that assumes the
// edge case can't happen, then hopefully this test will demand attention by
// either triggering an ICE, or needing its MIR to be re-blessed.

// EMIT_MIR array_non_capture.prefix_only-{closure#0}.built.after.mir
fn prefix_only() -> u32 {
    let arr = [1, 2, 3];
    let closure = || match arr {
        [_, _, _] => 101u32,
    };
    closure()
}

// EMIT_MIR array_non_capture.prefix_slice_only-{closure#0}.built.after.mir
fn prefix_slice_only() -> u32 {
    let arr = [1, 2, 3];
    let closure = || match arr {
        [_, ..] => 102u32,
    };
    closure()
}

// EMIT_MIR array_non_capture.prefix_slice_suffix-{closure#0}.built.after.mir
fn prefix_slice_suffix() -> u32 {
    let arr = [1, 2, 3];
    let closure = || match arr {
        [_, .., _] => 103u32,
    };
    closure()
}
