//@ compile-flags: -Z annotate-moves=8 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

// EMIT_MIR move_return.test_move.AnnotateMoves.after.mir
pub fn test_move(s: LargeStruct) -> LargeStruct {
    // CHECK-LABEL: fn test_move(
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    s
}
