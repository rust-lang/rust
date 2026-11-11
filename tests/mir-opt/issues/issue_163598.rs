// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ skip-filecheck
//@ compile-flags: -Z mir-opt-level=3
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Regression test for #163598. If the output has changed significantly (for example, no longer
// being mir-inlined), this may be acceptable if the final optimized assembly for `<=`
// on `[(T, U)]` hasn't regressed.

#![feature(partial_ord_chaining_methods)]

// EMIT_MIR issue_163598.chain_generic.runtime-optimized.after.mir
fn chain_generic<T: PartialOrd, U: PartialOrd>(
    l: &(T, U),
    r: &(T, U),
) -> core::ops::ControlFlow<bool> {
    PartialOrd::__chaining_less(l, r)
}
