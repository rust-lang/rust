//@ compile-flags: -Z annotate-moves=8 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct SmallStruct {
    pub data: u32, // 4 bytes
}

// EMIT_MIR small_move.test_small_move.AnnotateMoves.after.mir
pub fn test_small_move(s: SmallStruct) -> SmallStruct {
    // CHECK-LABEL: fn test_small_move(
    // Small types should NOT be annotated
    // CHECK-NOT: core::profiling::compiler_move
    // CHECK-NOT: core::profiling::compiler_copy
    s
}
