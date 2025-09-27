//@ compile-flags: -Z instrument-moves -Z instrument-moves-size-limit=8 -C debuginfo=full
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

#[derive(Clone)]
pub struct SmallStruct {
    pub data: u32, // 4 bytes
}

// EMIT_MIR basic.test_move.InstrumentMoves.after.mir
pub fn test_move(s: LargeStruct) -> LargeStruct {
    // CHECK-LABEL: fn test_move(
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    s
}

// EMIT_MIR basic.test_copy_field.InstrumentMoves.after.mir
pub fn test_copy_field(s: &LargeStruct) -> [u64; 20] {
    // CHECK-LABEL: fn test_copy_field(
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_copy::<[u64; 20], 160>)
    s.data
}

// EMIT_MIR basic.test_small_move.InstrumentMoves.after.mir
pub fn test_small_move(s: SmallStruct) -> SmallStruct {
    // CHECK-LABEL: fn test_small_move(
    // Small types should NOT be instrumented
    // CHECK-NOT: core::profiling::compiler_move
    // CHECK-NOT: core::profiling::compiler_copy
    s
}

// EMIT_MIR basic.test_call_arg.InstrumentMoves.after.mir
pub fn test_call_arg(s: LargeStruct) {
    // CHECK-LABEL: fn test_call_arg(
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    helper(s);
}

#[inline(never)]
fn helper(_s: LargeStruct) {}

// EMIT_MIR basic.test_aggregate.InstrumentMoves.after.mir
pub fn test_aggregate() -> LargeStruct {
    // CHECK-LABEL: fn test_aggregate(
    // Struct initialization with Rvalue::Aggregate
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_copy::<[u64; 20], 160>)
    let data = [0u64; 20];
    LargeStruct { data }
}

// EMIT_MIR basic.test_match.InstrumentMoves.after.mir
pub fn test_match(opt: Option<LargeStruct>) -> LargeStruct {
    // CHECK-LABEL: fn test_match(
    // Move in match expression
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    match opt {
        Some(s) => s,
        None => LargeStruct { data: [0; 20] },
    }
}
