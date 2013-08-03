// Test that attempt to reborrow an `&mut` pointer in an aliasable
// location yields an error.
//
// Example from src/middle/borrowck/doc.rs

use std::util::swap;

fn foo(t0: & &mut int) {
    let t1 = t0;
    let p: &int = &**t0; //~ ERROR cannot borrow an `&mut` in a `&` pointer
    **t1 = 22; //~ ERROR cannot assign
}

fn foo3(t0: &mut &mut int) {
    let t1 = &mut *t0;
    let p: &int = &**t0; //~ ERROR cannot borrow
    **t1 = 22;
}

fn main() {
}
