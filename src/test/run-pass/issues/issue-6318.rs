// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub enum Thing {
    A(Box<Foo+'static>)
}

pub trait Foo {
    fn dummy(&self) { }
}

pub struct Struct;

impl Foo for Struct {}

pub fn main() {
    match Thing::A(box Struct as Box<Foo+'static>) {
        Thing::A(_a) => 0,
    };
}
