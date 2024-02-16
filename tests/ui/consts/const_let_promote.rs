//@ run-pass

use std::cell::Cell;

const X: Option<Cell<i32>> = None;

const Y: Option<Cell<i32>> = {
    let x = None;
    x
};

// Ensure that binding the final value of a `const` to a variable does not affect promotion.
#[allow(unused)]
fn main() {
    let x: &'static _ = &X;
    let y: &'static _ = &Y;
}
