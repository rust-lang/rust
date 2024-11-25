//@ run-fail
//@ check-run-results:attempt to negate with overflow
//@ ignore-emscripten no processes
//@ compile-flags: -C debug-assertions
#![allow(arithmetic_overflow)]

use std::num::NonZero;

fn main() {
    let _x = -NonZero::new(i8::MIN).unwrap();
}
