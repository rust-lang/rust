// skip-filecheck
// unit-test: InstCombine
#![crate_type = "lib"]

// EMIT_MIR ref_deref.ref_deref.InstCombine.diff
pub fn ref_deref<T: Copy>(t: T) -> T {
    let r = &t;
    *r
}
