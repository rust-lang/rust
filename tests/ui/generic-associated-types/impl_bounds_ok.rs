//@ check-pass

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
    type A<'x> = (&'x ()) where T: 'x;
    type B<'u, 'v> = (&'v &'u ()) where 'u: 'v;
    type C = String where Self: Clone + ToOwned;
}

fn main() {}
