// This test verifies that the -Z separate-spans flag works correctly
// with incremental compilation.
//
// The separate-spans flag causes span data to be stored in a separate
// .spans file rather than inline in the .rmeta file. This is important
// for Relink, Don't Rebuild (RDR) because span data often
// contains absolute file paths that break reproducibility.
//
// rpass1: Initial compilation without separate-spans
// rpass2: Recompile without changes - should reuse everything
// rpass3: Recompile auxiliary with separate-spans - tests span loading

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph -g
//@ aux-build: separate_spans_lib.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

// The main module should be reused in rpass2 since nothing changed.
#![rustc_partition_reused(module = "main", cfg = "rpass2")]

extern crate separate_spans_lib;

// Use the macro to generate a function - this tests macro expansion spans
separate_spans_lib::generate_fn!(generated_fortytwo, 42);

fn main() {
    let result = separate_spans_lib::inlined_with_span();
    assert_eq!(result, 3);

    let s1 = separate_spans_lib::generic_with_span(42u32);
    let s2 = separate_spans_lib::generic_with_span("hello");
    assert!(s1.contains("42"));
    assert!(s2.contains("hello"));

    let (a, b, c) = separate_spans_lib::multi_span_fn();
    assert_eq!(a + b + c, 6);

    assert_eq!(generated_fortytwo(), 42);
}
