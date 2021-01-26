#![allow(unconditional_panic, const_err)]

// run-fail
// error-pattern: attempt to divide by zero
// ignore-emscripten no processes

fn main() {
    let x = &(1 / (1 - 1));
}
