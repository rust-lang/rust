// run-fail
// error-pattern:thread 'main' panicked
// error-pattern:attempt to subtract with overflow
// ignore-emscripten no processes
// compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
