// check-pass

#![allow(incomplete_features)]
#![feature(generic_associated_types)]
#![feature(associated_type_defaults)]

trait Foo {
    type A<'a> where Self: 'a;
    type B<'a, 'b> where 'a: 'b;
    type C where Self: Clone;
}

#[derive(Clone)]
struct Fooy;

impl Foo for Fooy {
    type A<'a> = (&'a ());
    type B<'a: 'b, 'b> = (&'a(), &'b ());
    type C = String;
}

#[derive(Clone)]
struct Fooer<T>(T);

impl<T> Foo for Fooer<T> {
    type A<'x> where T: 'x = (&'x ());
    type B<'u, 'v> where 'u: 'v = (&'v &'u ());
    type C where Self: Clone + ToOwned = String;
}

fn main() {}
