// skip-filecheck
// unit-test: InstCombine
#![crate_type = "lib"]

// EMIT_MIR ptr_cast.ptr_cast.InstCombine.diff
pub fn ptr_cast(p: *const u8) -> *mut () {
    p as *mut u8 as *mut ()
}
