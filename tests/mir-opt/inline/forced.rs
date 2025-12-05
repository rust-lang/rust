// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 --crate-type=lib
#![feature(rustc_attrs)]

#[rustc_force_inline]
pub fn callee_forced() {}

// EMIT_MIR forced.caller.ForceInline.diff
pub fn caller() {
    callee_forced();
    // CHECK-LABEL: fn caller(
    // CHECK: (inlined callee_forced)
}
