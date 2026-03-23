//@ build-pass

pub trait Foo {
    type FooAssoc;
}

pub struct Bar<F: Foo> {
    id: F::FooAssoc
}

pub struct Baz;

impl Foo for Baz {
    type FooAssoc = usize;
}

static mut MY_FOO: Bar<Baz> = Bar { id: 0 };

fn main() {}
