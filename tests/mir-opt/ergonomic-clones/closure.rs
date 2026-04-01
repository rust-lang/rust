#![crate_type = "lib"]
#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

pub fn ergonomic_clone_closure_move() -> String {
    // CHECK-LABEL: fn ergonomic_clone_closure_move(
    // CHECK: _0 = move (_1.0: std::string::String);
    // CHECK-NOT: <String as Clone>::clone
    let s = String::from("hi");

    let cl = use || s;
    cl()
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

pub fn ergonomic_clone_closure_use_cloned() -> Foo {
    // CHECK-LABEL: fn ergonomic_clone_closure_use_cloned(
    // CHECK: <Foo as Clone>::clone
    let f = Foo;

    let f1 = use || f;

    let f2 = use || f;

    f
}

pub fn ergonomic_clone_closure_copy() -> i32 {
    // CHECK-LABEL: fn ergonomic_clone_closure_copy(
    // CHECK: _0 = copy ((*_1).0: i32);
    // CHECK-NOT: <i32 as Clone>::clone
    let i = 1;

    let i1 = use || i;

    let i2 = use || i;

    i
}

pub fn ergonomic_clone_closure_use_cloned_generics<T: UseCloned>(f: T) -> T {
    // CHECK-LABEL: fn ergonomic_clone_closure_use_cloned_generics(
    // CHECK: <T as Clone>::clone
    let f1 = use || f;

    let f2 = use || f;

    f
}
