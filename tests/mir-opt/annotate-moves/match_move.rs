//@ compile-flags: -Z annotate-moves=8 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

// EMIT_MIR match_move.test_match.AnnotateMoves.after.mir
pub fn test_match(opt: Option<LargeStruct>) -> LargeStruct {
    // CHECK-LABEL: fn test_match(
    // Move in match expression
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    match opt {
        Some(s) => s,
        None => LargeStruct { data: [0; 20] },
    }
}
