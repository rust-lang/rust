//@ run-fail
//@ error-pattern:attempt to negate with overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions
#![allow(arithmetic_overflow)]

use std::num::NonZero;

fn main() {
    let _x = -NonZero::new(i8::MIN).unwrap();
}
