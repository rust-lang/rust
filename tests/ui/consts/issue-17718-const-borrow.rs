use std::cell::UnsafeCell;

const A: UnsafeCell<usize> = UnsafeCell::new(1);
const B: &'static UnsafeCell<usize> = &A;
//~^ ERROR: cannot refer to interior mutable

struct C { a: UnsafeCell<usize> }
const D: C = C { a: UnsafeCell::new(1) };
const E: &'static UnsafeCell<usize> = &D.a;
//~^ ERROR: cannot refer to interior mutable
const F: &'static C = &D;
//~^ ERROR: cannot refer to interior mutable

fn main() {}
