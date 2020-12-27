#![feature(const_refs_to_cell)]

use std::cell::UnsafeCell;

const A: UnsafeCell<usize> = UnsafeCell::new(1);
const B: &'static UnsafeCell<usize> = &A;
//~^ ERROR: may contain interior mutability

struct C { a: UnsafeCell<usize> }
const D: C = C { a: UnsafeCell::new(1) };
const E: &'static UnsafeCell<usize> = &D.a;
//~^ ERROR: may contain interior mutability
const F: &'static C = &D;
//~^ ERROR: may contain interior mutability

fn main() {}
