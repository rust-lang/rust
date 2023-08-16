// run-fail
//@error-in-other-file:thread 'main' panicked at 'attempt to negate with overflow'
//@ignore-target-emscripten no processes
//@compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = -i8::MIN;
}
