// skip-filecheck
// compile-flags: -Zmir-opt-level=2 -Zinline-mir
// no-debug

#![crate_type = "lib"]

// Theoretically, Vec::as_ptr could be implemented with a single assignment,
// and a long projection. This tests tracks how close we are to that, without
// breaking the companion Vec::as_ptr codegen test.

// EMIT_MIR vec_as_ptr.as_ptr.InstCombine.diff
// EMIT_MIR vec_as_ptr.as_ptr.PreCodegen.after.mir
pub fn as_ptr(v: &Vec<i32>) -> *const i32 {
    v.as_ptr()
}
