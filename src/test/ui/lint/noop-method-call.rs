// check-pass

#![allow(unused)]
#![warn(noop_method_call)]

use std::borrow::Borrow;
use std::ops::Deref;

struct PlainType<T>(T);

#[derive(Clone)]
struct CloneType<T>(T);

fn main() {
    let non_clone_type_ref = &PlainType(1u32);
    let non_clone_type_ref_clone: &PlainType<u32> = non_clone_type_ref.clone();
    //~^ WARNING call to `.clone()` on a reference in this situation does nothing

    let clone_type_ref = &CloneType(1u32);
    let clone_type_ref_clone: CloneType<u32> = clone_type_ref.clone();

    // Calling clone on a double reference doesn't warn since the method call itself
    // peels the outer reference off
    let clone_type_ref = &&CloneType(1u32);
    let clone_type_ref_clone: &CloneType<u32> = clone_type_ref.clone();

    let non_deref_type = &PlainType(1u32);
    let non_deref_type_deref: &PlainType<u32> = non_deref_type.deref();
    //~^ WARNING call to `.deref()` on a reference in this situation does nothing

    // Dereferencing a &&T does not warn since it has collapsed the double reference
    let non_deref_type = &&PlainType(1u32);
    let non_deref_type_deref: &PlainType<u32> = non_deref_type.deref();

    let non_borrow_type = &PlainType(1u32);
    let non_borrow_type_borrow: &PlainType<u32> = non_borrow_type.borrow();
    //~^ WARNING call to `.borrow()` on a reference in this situation does nothing

    // Borrowing a &&T does not warn since it has collapsed the double reference
    let non_borrow_type = &&PlainType(1u32);
    let non_borrow_type_borrow: &PlainType<u32> = non_borrow_type.borrow();

    let xs = ["a", "b", "c"];
    let _v: Vec<&str> = xs.iter().map(|x| x.clone()).collect(); // ok, but could use `*x` instead
}

fn generic<T>(non_clone_type: &PlainType<T>) {
    non_clone_type.clone();
}

fn non_generic(non_clone_type: &PlainType<u32>) {
    non_clone_type.clone();
    //~^ WARNING call to `.clone()` on a reference in this situation does nothing
}
