use std::cell::UnsafeCell;

const A: UnsafeCell<usize> = UnsafeCell::new(1);
const B: &'static UnsafeCell<usize> = &A;
//~^ ERROR: cannot borrow a constant which may contain interior mutability

struct C { a: UnsafeCell<usize> }
const D: C = C { a: UnsafeCell::new(1) };
const E: &'static UnsafeCell<usize> = &D.a;
//~^ ERROR: cannot borrow a constant which may contain interior mutability
const F: &'static C = &D;
//~^ ERROR: cannot borrow a constant which may contain interior mutability

fn main() {}
