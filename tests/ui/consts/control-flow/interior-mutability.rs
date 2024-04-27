// Ensure that *any* assignment to the return place of a value with interior mutability
// disqualifies it from promotion.

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

const Z: Option<Cell<i32>> = {
    let mut z = None;
    let mut i = 0;
    while i < 10 {
        if i == 8 {
            z = Some(Cell::new(4));
        }

        if i == 9 {
            z = None;
        }

        i += 1;
    }
    z
};

fn main() {
    let x: &'static _ = &X; //~ ERROR temporary value dropped while borrowed
    let y: &'static _ = &Y; //~ ERROR temporary value dropped while borrowed
    let z: &'static _ = &Z; //~ ERROR temporary value dropped while borrowed
}
