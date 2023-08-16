// run-fail
//@error-in-other-file:thread 'main' panicked at 'attempt to subtract with overflow'
//@ignore-target-emscripten no processes
//@compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
