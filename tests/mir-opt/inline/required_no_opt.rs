// EMIT_MIT_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Copt-level=0 --crate-type=lib
#![feature(required_inlining)]

#[inline(required)]
pub fn callee_required() {}

#[inline(must)]
pub fn callee_must() {}

// EMIT_MIR required_no_opt.caller.Inline.diff
pub fn caller() {
    callee_required();
    callee_must();
    // CHECK-LABEL: fn caller(
    // CHECK: (inlined callee_required)
    // CHECK: (inlined callee_must)
}
