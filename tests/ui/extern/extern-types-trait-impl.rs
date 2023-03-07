// run-pass
#![allow(dead_code)]
// Test that traits can be implemented for extern types.
#![feature(extern_types)]

extern "C" {
    type A;
}

trait Foo {
    fn foo(&self) {}
}

impl Foo for A {
    fn foo(&self) {}
}

fn assert_foo<T: ?Sized + Foo>() {}

fn use_foo<T: ?Sized + Foo>(x: &dyn Foo) {
    x.foo();
}

fn main() {
    assert_foo::<A>();
}
