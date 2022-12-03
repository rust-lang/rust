// skip-filecheck
// unit-test: InstCombine
#![crate_type = "lib"]

// EMIT_MIR ref_addressof.ref_addressof.InstCombine.diff
pub fn ref_addressof<T>(t: T) {
    let r = &t;
    let ptr = std::ptr::addr_of!(*r);
    drop(ptr);
}
