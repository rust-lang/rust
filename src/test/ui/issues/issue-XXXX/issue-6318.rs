// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub enum Thing {
    A(Box<dyn Foo+'static>)
}

pub trait Foo {
    fn dummy(&self) { }
}

pub struct Struct;

impl Foo for Struct {}

pub fn main() {
    match Thing::A(box Struct as Box<dyn Foo + 'static>) {
        Thing::A(_a) => 0,
    };
}
