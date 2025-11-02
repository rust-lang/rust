//@ compile-flags: -O

#![crate_type = "lib"]

// EMIT_MIR two_unwrap_unchecked.two_unwrap_unchecked.GVN.diff
// EMIT_MIR two_unwrap_unchecked.two_unwrap_unchecked.PreCodegen.after.mir
pub fn two_unwrap_unchecked(v: &Option<i32>) -> i32 {
    // CHECK-LABEL: fn two_unwrap_unchecked(
    // CHECK: [[DEREF_V:_.*]] = copy (*_1);
    // CHECK: [[V1V2:_.*]] = copy (([[DEREF_V]] as Some).0: i32);
    // CHECK: _0 = Add(copy [[V1V2]], copy [[V1V2]]);
    let v1 = unsafe { v.unwrap_unchecked() };
    let v2 = unsafe { v.unwrap_unchecked() };
    v1 + v2
}
