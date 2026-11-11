// https://github.com/rust-lang/rust/issues/6318
//@ run-pass

pub enum Thing {
    A(Box<dyn Foo+'static>)
}

pub trait Foo {
    fn dummy(&self) { }
}

pub struct Struct;

impl Foo for Struct {}

pub fn main() {
    match Thing::A(Box::new(Struct) as Box<dyn Foo + 'static>) {
        Thing::A(_a) => 0,
    };
}
