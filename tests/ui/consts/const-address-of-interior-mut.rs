#![feature(raw_ref_op)]

use std::cell::Cell;

const A: () = { let x = Cell::new(2); &raw const x; };      //~ ERROR interior mutability

static B: () = { let x = Cell::new(2); &raw const x; };     //~ ERROR interior mutability

static mut C: () = { let x = Cell::new(2); &raw const x; }; //~ ERROR interior mutability

const fn foo() {
    let x = Cell::new(0);
    let y = &raw const x;                                   //~ ERROR interior mutability
}

fn main() {}
