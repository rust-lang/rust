// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// skip-filecheck
//@ test-mir-pass: Inline
//@ edition: 2021
//@ compile-flags: -Zinline-mir --crate-type=lib

// EMIT_MIR inline_double_cycle.a.Inline.diff
// EMIT_MIR inline_double_cycle.b.Inline.diff

#![feature(fn_traits)]

#[inline]
pub fn a() {
    FnOnce::call_once(a, ());
    FnOnce::call_once(b, ());
}

#[inline]
pub fn b() {
    FnOnce::call_once(b, ());
    FnOnce::call_once(a, ());
}
