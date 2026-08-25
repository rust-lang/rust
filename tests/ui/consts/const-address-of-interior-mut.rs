//@check-pass
use std::cell::Cell;

const A: () = { let x = Cell::new(2); &raw const x; };

static B: () = { let x = Cell::new(2); &raw const x; };

static mut C: () = { let x = Cell::new(2); &raw const x; };

const fn foo() {
    let x = Cell::new(0);
    let y = &raw const x;
}

fn main() {}
