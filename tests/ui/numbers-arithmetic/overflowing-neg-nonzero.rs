// run-fail
//@error-in-other-file:thread 'main' panicked at 'attempt to negate with overflow'
//@ignore-target-emscripten no processes
//@compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

use std::num::NonZeroI8;

fn main() {
    let _x = -NonZeroI8::new(i8::MIN).unwrap();
}
