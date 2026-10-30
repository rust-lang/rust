//@ compile-flags: -C no-prepopulate-passes -Copt-level=0 -Zmir-opt-level=0

#![crate_type = "lib"]
#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo;

impl UseCloned for Foo {}

pub fn ergonomic_clone_closure_use_cloned() -> Foo {
    let f = Foo;

    // CHECK: ; {{(call|invoke)}} <closure::Foo as core::clone::Clone>::clone
    let f1 = || f.use;

    // CHECK: ; {{(call|invoke)}} <closure::Foo as core::clone::Clone>::clone
    let f2 = || f.use;

    f
}

pub fn ergonomic_clone_closure_copy() -> i32 {
    let i = 1;

    // CHECK-NOT: ; {{(call|invoke)}} core::clone::impls::<impl core::clone::Clone for i32>::clone
    let i1 = || i.use;

    // CHECK-NOT: ; {{(call|invoke)}} core::clone::impls::<impl core::clone::Clone for i32>::clone
    let i2 = || i.use;

    i
}

pub fn ergonomic_clone_closure_use_cloned_generics<T: UseCloned>(f: T) -> T {
    // CHECK-NOT: ; {{(call|invoke)}} core::clone::impls::<impl core::clone::Clone for i32>::clone
    let f1 = || f.use;

    // CHECK-NOT: ; {{(call|invoke)}} core::clone::impls::<impl core::clone::Clone for i32>::clone
    let f2 = || f.use;

    f
}
