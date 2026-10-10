//@ run-pass
#![allow(dead_code)]
// Test that traits can be implemented for extern types.
#![feature(extern_types, sized_hierarchy)]
use std::marker::PointeeSized;

extern "C" {
    type A;
}

trait Foo: PointeeSized {
    fn foo(&self) {}
}

impl Foo for A {
    fn foo(&self) {}
}

fn assert_foo<T: PointeeSized + Foo>() {}

fn use_foo<T: PointeeSized + Foo>(x: &dyn Foo) {
    x.foo();
}

fn main() {
    assert_foo::<A>();
}
