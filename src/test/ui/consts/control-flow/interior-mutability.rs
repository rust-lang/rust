// Ensure that *any* assignment to the return place of a value with interior mutability
// disqualifies it from promotion.

#![feature(const_if_match)]

use std::cell::Cell;

const X: Option<Cell<i32>> = {
    let mut x = None;
    if false {
        x = Some(Cell::new(4));
    }
    x
};

const Y: Option<Cell<i32>> = {
    let mut y = Some(Cell::new(4));
    if true {
        y = None;
    }
    y
};

fn main() {
    let x: &'static _ = &X; //~ ERROR temporary value dropped while borrowed
    let y: &'static _ = &Y; //~ ERROR temporary value dropped while borrowed
}
