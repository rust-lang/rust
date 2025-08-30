//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck
#![feature(never_type)]
#![allow(unreachable_code)]

// EMIT_MIR eq_never_type._f.built.after.mir
fn _f(a: !, b: !) {
    // Both arguments must be references (i.e. == should auto-borrow/coerce-to-ref both arguments)
    // (this previously was buggy due to `NeverToAny` coercion incorrectly throwing out other
    // coercions)
    a == b;
}

fn main() {}
