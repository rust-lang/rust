//@ compile-flags: -O

#![crate_type = "lib"]

// EMIT_MIR two_unwrap_unchecked.two_unwrap_unchecked.GVN.diff
// EMIT_MIR two_unwrap_unchecked.two_unwrap_unchecked.PreCodegen.after.mir
pub fn two_unwrap_unchecked(v: &Option<i32>) -> i32 {
    // CHECK-LABEL: fn two_unwrap_unchecked(
    // CHECK: [[DEREF_V:_.*]] = copy (*_1);
    // CHECK: _0 = Add(copy (([[DEREF_V]] as Some).0: i32), copy (([[DEREF_V]] as Some).0: i32));
    let v1 = unsafe { v.unwrap_unchecked() };
    let v2 = unsafe { v.unwrap_unchecked() };
    v1 + v2
}
