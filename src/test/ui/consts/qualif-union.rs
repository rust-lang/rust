// Checks that unions use type based qualification. Regression test for issue #90268.
#![feature(untagged_unions)]
use std::cell::Cell;

union U { i: u32, c: Cell<u32> }

const C1: Cell<u32> = {
    unsafe { U { c: Cell::new(0) }.c }
};

const C2: Cell<u32> = {
    unsafe { U { i : 0 }.c }
};

const C3: Cell<u32> = {
    let mut u = U { i: 0 };
    u.i = 1;
    unsafe { u.c }
};

const C4: U = U { i: 0 };

const C5: [U; 1] = [U {i : 0}; 1];

fn main() {
    // Interior mutability should prevent promotion.
    let _: &'static _ = &C1; //~ ERROR temporary value dropped while borrowed
    let _: &'static _ = &C2; //~ ERROR temporary value dropped while borrowed
    let _: &'static _ = &C3; //~ ERROR temporary value dropped while borrowed
    let _: &'static _ = &C4; //~ ERROR temporary value dropped while borrowed
    let _: &'static _ = &C5; //~ ERROR temporary value dropped while borrowed
}
