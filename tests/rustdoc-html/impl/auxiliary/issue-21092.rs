//@ compile-flags: -Cmetadata=aux

pub trait Foo {
    type Bar;
    fn foo(&self) {}
}

pub struct Bar;

impl Foo for Bar {
    type Bar = i32;
}
