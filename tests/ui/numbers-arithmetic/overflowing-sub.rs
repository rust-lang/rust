//@ run-fail
//@ check-run-results:thread 'main' panicked
//@ check-run-results:attempt to subtract with overflow
//@ ignore-emscripten no processes
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
