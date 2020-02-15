// Regression test for #39618, shouldn't crash.
// check-pass

#![feature(specialization)]

trait Foo {
    fn foo(&self);
}

trait Bar {
    fn bar(&self);
}

impl<T> Bar for T where T: Foo {
    fn bar(&self) {}
}

impl<T> Foo for T where T: Bar {
    fn foo(&self) {}
}

impl Foo for u64 {}

fn main() {}
