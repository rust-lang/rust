// run-fail
// error-pattern:index out of bounds
// ignore-emscripten no processes

#![warn(unconditional_panic)]

use std::mem::size_of;

fn main() {
    let xs = [1, 2, 3];
    xs[usize::MAX / size_of::<isize>() + 1]; //~ WARN: this operation will panic at runtime
}
