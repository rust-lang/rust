use std::cell::UnsafeCell;

const A: UnsafeCell<usize> = UnsafeCell::new(1);
const B: &'static UnsafeCell<usize> = &A;
//~^ ERROR: borrow to an interior mutable value

struct C { a: UnsafeCell<usize> }
const D: C = C { a: UnsafeCell::new(1) };
const E: &'static UnsafeCell<usize> = &D.a;
//~^ ERROR: borrow to an interior mutable value
const F: &'static C = &D;
//~^ ERROR: borrow to an interior mutable value

fn main() {}
