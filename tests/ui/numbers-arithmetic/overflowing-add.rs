//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:attempt to add with overflow
//@ compile-flags: -C debug-assertions
//@ ignore-emscripten no processes

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 200u8 + 200u8 + 200u8;
}
