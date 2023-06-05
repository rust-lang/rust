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

    let clone_type_ref = &&CloneType(1u32);
    let clone_type_ref_clone: &CloneType<u32> = clone_type_ref.clone();
    //~^ WARNING using `.clone()` on a double reference, which returns `&CloneType<u32>`

    let non_deref_type = &PlainType(1u32);
    let non_deref_type_deref: &PlainType<u32> = non_deref_type.deref();
    //~^ WARNING call to `.deref()` on a reference in this situation does nothing

    let non_deref_type = &&PlainType(1u32);
    let non_deref_type_deref: &PlainType<u32> = non_deref_type.deref();
    //~^ WARNING using `.deref()` on a double reference, which returns `&PlainType<u32>`

    let non_borrow_type = &PlainType(1u32);
    let non_borrow_type_borrow: &PlainType<u32> = non_borrow_type.borrow();
    //~^ WARNING call to `.borrow()` on a reference in this situation does nothing

    // Borrowing a &&T does not warn since it has collapsed the double reference
    let non_borrow_type = &&PlainType(1u32);
    let non_borrow_type_borrow: &PlainType<u32> = non_borrow_type.borrow();

    let xs = ["a", "b", "c"];
    let _v: Vec<&str> = xs.iter().map(|x| x.clone()).collect(); // could use `*x` instead
    //~^ WARNING using `.clone()` on a double reference, which returns `&str`
}

fn generic<T>(non_clone_type: &PlainType<T>) {
    non_clone_type.clone();
    //~^ WARNING call to `.clone()` on a reference in this situation does nothing
}

fn non_generic(non_clone_type: &PlainType<u32>) {
    non_clone_type.clone();
    //~^ WARNING call to `.clone()` on a reference in this situation does nothing
}
