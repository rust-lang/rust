// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 --crate-type=lib
//@ edition: 2021
#![feature(rustc_attrs)]

#[rustc_force_inline]
pub fn callee_forced() {}

// EMIT_MIR forced_async.caller.ForceInline.diff
async fn caller() {
    callee_forced();
    // CHECK-LABEL: fn caller(
    // CHECK: (inlined callee_forced)
}
