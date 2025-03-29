// skip-filecheck

#![crate_type = "lib"]

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

// EMIT_MIR closure.ergonomic_clone_closure_move.nll.0.mir
pub fn ergonomic_clone_closure_move() -> String {
    let s = String::from("hi");

    let cl = use || s;
    cl()
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

// EMIT_MIR closure.ergonomic_clone_closure_use_cloned.nll.0.mir
pub fn ergonomic_clone_closure_use_cloned() -> Foo {
    let f = Foo;

    let f1 = use || f;

    let f2 = use || f;

    f
}

// EMIT_MIR closure.ergonomic_clone_closure_copy.nll.0.mir
pub fn ergonomic_clone_closure_copy() -> i32 {
    let i = 1;

    let i1 = use || i;

    let i2 = use || i;

    i
}

// EMIT_MIR closure.ergonomic_clone_closure_use_cloned_generics.nll.0.mir
pub fn ergonomic_clone_closure_use_cloned_generics<T: UseCloned>(f: T) -> T {
    let f1 = use || f;

    let f2 = use || f;

    f
}
