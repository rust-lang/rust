#![feature(cmp_splat)]

// Comparing the MIR of a splat-compatible function with it's direct counterpart.
// This test is to ensure MIR optimizations are able to remove the added layers
// of indirection.

// EMIT_MIR cmp_splat.min_direct_u8.runtime-optimized.after.mir
pub fn min_direct_u8(x: u8, y: u8) -> u8 {
    // CHECK-LABEL: fn min_direct_u8(
    // CHECK: debug x => _1;
    // CHECK: debug y => _2;
    // CHECK: bb0: {
    // CHECK-NEXT: _0 = integer_min::<u8>(move _1, move _2) -> [return: bb1, unwind unreachable];
    // CHECK-NEXT: }
    // CHECK: bb1: {
    // CHECK-NEXT: return;
    // CHECK-NEXT: }
    std::cmp::min(x, y)
}

// EMIT_MIR cmp_splat.min_splatted_u8.runtime-optimized.after.mir
pub fn min_splatted_u8(x: u8, y: u8) -> u8 {
    // CHECK-LABEL: fn min_splatted_u8(
    // CHECK: debug x => _1;
    // CHECK: debug y => _2;
    // CHECK: bb0: {
    // CHECK-NEXT: _0 = integer_min::<u8>(move _1, move _2) -> [return: bb1, unwind unreachable];
    // CHECK-NEXT: }
    // CHECK: bb1: {
    // CHECK-NEXT: return;
    // CHECK-NEXT: }
    std::cmp::smallest(x, y)
}
