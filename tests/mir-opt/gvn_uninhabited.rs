//@ test-mir-pass: GVN
//@ compile-flags: -O
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// skip-filecheck

#![feature(never_type)]

#[derive(Copy, Clone)]
pub enum E {
    A(!, u32),
}

pub union U {
    i: u32,
    e: E,
}

// EMIT_MIR gvn_uninhabited.f.GVN.diff
pub const fn f() -> u32 {
    let E::A(_, i) = unsafe { (&U { i: 0 }).e };
    i
}

fn main() {}
