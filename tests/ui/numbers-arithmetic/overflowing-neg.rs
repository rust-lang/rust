// run-fail
// error-pattern:thread 'main' panicked at 'attempt to negate with overflow'
// ignore-emscripten no processes
// compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = -i8::MIN;
}
