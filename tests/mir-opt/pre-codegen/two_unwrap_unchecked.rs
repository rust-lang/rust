// skip-filecheck
//@ compile-flags: -O

#![crate_type = "lib"]

// EMIT_MIR two_unwrap_unchecked.two_unwrap_unchecked.GVN.diff
// EMIT_MIR two_unwrap_unchecked.two_unwrap_unchecked.PreCodegen.after.mir
pub fn two_unwrap_unchecked(v: &Option<i32>) -> i32 {
    let v1 = unsafe { v.unwrap_unchecked() };
    let v2 = unsafe { v.unwrap_unchecked() };
    v1 + v2
}
