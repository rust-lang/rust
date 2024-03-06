//@ run-fail
//@ error-pattern:attempt to negate with overflow
//@ ignore-emscripten no processes
//@ compile-flags: -C debug-assertions
#![allow(arithmetic_overflow)]
#![feature(generic_nonzero)]

use std::num::NonZero;

fn main() {
    let _x = -NonZero::new(i8::MIN).unwrap();
}
