//@ check-fail

#![feature(associated_type_defaults)]

trait Foo {
    type A<'a> where Self: 'a;
}

struct Fooy;

impl Foo for Fooy {
    type A<'a> = &'a ();
}

#[derive(Clone)]
struct Fooer<T>(T);

impl<T> Foo for Fooer<T> {
    type A<'x> = &'x () where T: 'x;
}

fn f(_arg : Box<dyn for<'a> Foo<A<'a> = &'a ()>>) {}
//~^ the trait `Foo` cannot be made into an object


fn main() {
  let foo = Fooer(5);
  f(Box::new(foo));
  //~^ the trait `Foo` cannot be made into an object
  //~| the trait `Foo` cannot be made into an object
}
