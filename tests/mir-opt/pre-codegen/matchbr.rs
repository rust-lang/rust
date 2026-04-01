#![crate_type = "lib"]

// EMIT_MIR matchbr.match1.PreCodegen.after.mir
pub fn match1(c: bool, v1: i32, v2: i32) -> i32 {
    // CHECK-LABEL: fn match1(
    // CHECK: bb0:
    // CHECK-NEXT: _0 = Sub
    // CHECK-NEXT: return;
    if c { v1 - v2 } else { v1 - v2 }
}
