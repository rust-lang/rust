// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-opt-level=0

// Verify that matching against a constant array pattern produces a single
// `PartialEq::eq` call rather than element-by-element comparisons.

#![crate_type = "lib"]

// EMIT_MIR aggregate_array_eq.array_match.built.after.mir
pub fn array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn array_match(
    // CHECK: <[u8; 4] as PartialEq>::eq
    // CHECK-NOT: switchInt(copy _1[
    matches!(x, [1, 2, 3, 4])
}
