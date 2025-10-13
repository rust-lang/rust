//@ compile-flags: -Z annotate-moves=8 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

// EMIT_MIR aggregate.test_aggregate.AnnotateMoves.after.mir
pub fn test_aggregate() -> LargeStruct {
    // CHECK-LABEL: fn test_aggregate(
    // Struct initialization with Rvalue::Aggregate
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_copy::<[u64; 20], 160>)
    let data = [0u64; 20];
    LargeStruct { data }
}
