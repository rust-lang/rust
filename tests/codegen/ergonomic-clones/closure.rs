//@ compile-flags: -C no-prepopulate-passes -Copt-level=0 -Zmir-opt-level=0

#![crate_type = "lib"]

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

pub fn ergonomic_clone_closure_move() -> String {
    let s = String::from("hi");

    // CHECK-NOT: ; call core::clone::impls::<impl core::clone::Clone for String>::clone
    let cl = use || s;
    cl()
}

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

pub fn ergonomic_clone_closure_use_cloned() -> Foo {
    let f = Foo;

    // CHECK: ; call <closure::Foo as core::clone::Clone>::clone
    let f1 = use || f;

    // CHECK: ; call <closure::Foo as core::clone::Clone>::clone
    let f2 = use || f;

    f
}

pub fn ergonomic_clone_closure_copy() -> i32 {
    let i = 1;

    // CHECK-NOT: ; call core::clone::impls::<impl core::clone::Clone for i32>::clone
    let i1 = use || i;

    // CHECK-NOT: ; call core::clone::impls::<impl core::clone::Clone for i32>::clone
    let i2 = use || i;

    i
}

pub fn ergonomic_clone_closure_use_cloned_generics<T: UseCloned>(f: T) -> T {
    // CHECK-NOT: ; call core::clone::impls::<impl core::clone::Clone for i32>::clone
    let f1 = use || f;

    // CHECK-NOT: ; call core::clone::impls::<impl core::clone::Clone for i32>::clone
    let f2 = use || f;

    f
}
