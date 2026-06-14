#![feature(view_types)]
#![allow(unused)]

struct Foo {
    bar: usize,
    baz: usize,
}

impl Foo {
    fn not_a_field(&mut self.{ _ }, _: &mut Foo.{ _ }) {}
    //~^ ERROR expected identifier, found reserved identifier
    //~| ERROR expected identifier, found reserved identifier

    fn keyword(&mut self.{ where }, _: &mut Foo.{ for }) {}
    //~^ ERROR expected identifier, found keyword
    //~| ERROR expected identifier, found keyword

    fn no_comma(&mut self.{ bar baz }, _: &mut Foo.{ bar baz }) {}
    //~^ ERROR expected one of `,` or `}`, found `baz`
    //~| ERROR expected one of `,` or `}`, found `baz`
}

fn main() {}
