use std::cell::Cell;

const WRITE: () = unsafe {
    let x = Cell::new(0);
    let y = &x;
    //~^ ERROR interior mutability
    //~| HELP add `#![feature(const_refs_to_cell)]` to the crate attributes to enable
};

fn main() {}
