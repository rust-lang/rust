//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:attempt to multiply with overflow
//@ ignore-emscripten no processes
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let x = 200u8 * 4;
}
