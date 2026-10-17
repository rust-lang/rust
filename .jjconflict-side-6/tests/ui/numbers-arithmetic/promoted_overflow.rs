#![allow(arithmetic_overflow)]

//@ run-fail
//@ error-pattern: overflow
//@ compile-flags: -C overflow-checks=yes
// for some reason, fails to match error string on
// wasm32-unknown-unknown with stripped debuginfo and symbols,
// so don't strip it
//@ compile-flags:-Cstrip=none

fn main() {
    let x: &'static u32 = &(0u32 - 1);
}
