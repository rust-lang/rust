#![feature(view_types)]
#![allow(unused)]

struct Foo {
    bar: usize,
    baz: usize,
}

impl Foo {
    fn not_a_field(&mut self.{ _ }, _: &mut Foo.{ _ }) {}
    //~^ ERROR expected parameter name
    //~| ERROR expected one of
    //~| ERROR expected identifier

    fn keyword(&mut self.{ where }, _: &mut Foo.{ for }) {}
    //~^ ERROR expected parameter name
    //~| ERROR expected one of
    //~| ERROR expected identifier

    fn no_comma(&mut self.{ bar baz }, _: &mut Foo.{ bar baz }) {}
    //~^ ERROR expected parameter name
    //~| ERROR expected one of
    //~| ERROR expected one of
}

fn main() {}
