// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// Added after it stopped inlining in a nightly; see
// <https://github.com/rust-lang/rust/issues/123174>

// EMIT_MIR vec_deref.vec_deref_to_slice.PreCodegen.after.mir
pub fn vec_deref_to_slice(v: &Vec<u8>) -> &[u8] {
    v
}
