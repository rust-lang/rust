// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#![crate_type = "lib"]
#![feature(rustc_attrs)]

//@ compile-flags: -Zmir-opt-level=2 -Zinline-mir

#[inline]
#[rustc_no_mir_inline]
pub fn callee() {}

// EMIT_MIR rustc_no_mir_inline.caller.Inline.diff
// EMIT_MIR rustc_no_mir_inline.caller.PreCodegen.after.mir
pub fn caller() {
    // CHECK-LABEL: fn caller(
    // CHECK: callee()
    callee();
}
