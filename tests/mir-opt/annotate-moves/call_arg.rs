//@ compile-flags: -Z annotate-moves=8 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

// EMIT_MIR call_arg.test_call_arg.AnnotateMoves.after.mir
pub fn test_call_arg(s: LargeStruct) {
    // CHECK-LABEL: fn test_call_arg(
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    helper(s);
}

#[inline(never)]
fn helper(_s: LargeStruct) {}
