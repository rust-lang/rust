//! Regression test for https://github.com/rust-lang/rust/issues/10456

//@ check-pass

pub struct Foo;

pub trait Bar {
    fn bar(&self);
}

pub trait Baz {
    fn baz(&self) { }
}

impl<T: Baz> Bar for T {
    fn bar(&self) {}
}

impl Baz for Foo {}

pub fn foo(t: Box<Foo>) {
    t.bar(); // ~Foo doesn't implement Baz
    (*t).bar(); // ok b/c Foo implements Baz
}

fn main() {}
