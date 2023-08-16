// run-fail
//@error-in-other-file:thread 'main' panicked at 'attempt to multiply with overflow'
//@ignore-target-emscripten no processes
//@compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let x = 200u8 * 4;
}
