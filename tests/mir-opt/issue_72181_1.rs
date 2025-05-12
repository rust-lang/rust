// skip-filecheck
//@ compile-flags: -Z mir-opt-level=1
// Regression test for #72181, this ICE requires `-Z mir-opt-level=1` flags.

#![feature(never_type)]
#![allow(unused, invalid_value)]

enum Void {}

// EMIT_MIR issue_72181_1.f.built.after.mir
fn f(v: Void) -> ! {
    match v {}
}

// EMIT_MIR issue_72181_1.main.built.after.mir
fn main() {
    let v: Void = unsafe { std::mem::transmute::<(), Void>(()) };

    f(v);
}
