use std::cell::UnsafeCell;

const A: UnsafeCell<usize> = UnsafeCell::new(1);
const B: &'static UnsafeCell<usize> = &A;
//~^ ERROR: interior mutable shared borrows of lifetime-extended temporaries

struct C { a: UnsafeCell<usize> }
const D: C = C { a: UnsafeCell::new(1) };
const E: &'static UnsafeCell<usize> = &D.a;
//~^ ERROR: interior mutable shared borrows of lifetime-extended temporaries
const F: &'static C = &D;
//~^ ERROR: interior mutable shared borrows of lifetime-extended temporaries

fn main() {}
