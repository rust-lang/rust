//@ compile-flags: -Z annotate-moves=8 -Copt-level=0 -g

#![crate_type = "lib"]

#[derive(Clone, Copy)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

// This test verifies that when passing arguments to functions, the actual CALL instruction
// does not have the compiler_move debug scope, even though the argument itself might be
// annotated with compiler_move in MIR.
//
// Note: On most ABIs, large structs are passed by pointer even when written as "by value",
// so there may not be an actual memcpy operation to attach compiler_move to. This test
// mainly verifies that IF debug info is emitted, the call itself uses the source location.

// CHECK-LABEL: call_arg_scope::test_call_with_move
pub fn test_call_with_move(s: LargeStruct) {
    // The key test: the call instruction should reference the source line (line 22),
    // NOT a compiler_move scope.
    helper(s);
}

// Find the call instruction and verify its debug location
// CHECK: call {{.*}}@{{.*}}helper{{.*}}({{.*}}), !dbg ![[CALL_LOC:[0-9]+]]

// Verify that the call's debug location points to line 22 (the actual source line)
// and NOT to a scope with inlinedAt referencing compiler_move
// CHECK: ![[CALL_LOC]] = !DILocation(line: 22,

#[inline(never)]
fn helper(_s: LargeStruct) {}
