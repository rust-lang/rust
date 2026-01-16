// This test verifies that spans embedded in crate metadata are handled
// correctly for incremental compilation and reproducibility.
//
// The test exercises:
// 1. Cross-crate inlined function spans
// 2. Macro expansion spans from external crates
// 3. Trait implementation spans
// 4. Panic location spans in inlined code
//
// rpass1: Initial compilation
// rpass2: Recompile with no changes - should reuse everything
// rpass3: Recompile with path remapping - tests span normalization

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph -g
//@ aux-build: spans_lib.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

// In rpass2, we should reuse the codegen for main since nothing changed.
// In rpass3, path remapping in the auxiliary crate may affect spans.
#![rustc_partition_reused(module = "main", cfg = "rpass2")]

extern crate spans_lib;

use spans_lib::{SpannedStruct, SpannedTrait};

fn main() {
    let _: u32 = spans_lib::generic_fn();
    let _: String = spans_lib::generic_fn();

    let s = SpannedStruct { field1: 42, field2: String::from("test") };
    let _ = s.field1;

    let x = spans_lib::span_macro!(41);
    assert_eq!(x, 42);

    let val: u32 = 21;
    let result = val.process();
    assert_eq!(result, 42);

    let _ = spans_lib::might_panic(1);
}
