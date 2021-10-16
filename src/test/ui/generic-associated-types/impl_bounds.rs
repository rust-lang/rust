#![feature(generic_associated_types)]
#![feature(associated_type_defaults)]

trait Foo {
    type A<'a> where Self: 'a;
    type B<'a, 'b> where 'a: 'b;
    type C where Self: Clone;
    fn d() where Self: Clone;
}

#[derive(Copy, Clone)]
struct Fooy<T>(T);

impl<T> Foo for Fooy<T> {
    type A<'a> where Self: 'static = (&'a ());
    //~^ ERROR `impl` associated type
    type B<'a, 'b> where 'b: 'a = (&'a(), &'b ());
    //~^ ERROR `impl` associated type
    //~| ERROR lifetime bound not satisfied
    type C where Self: Copy = String;
    //~^ ERROR the trait bound `T: Copy` is not satisfied
    fn d() where Self: Copy {}
    //~^ ERROR the trait bound `T: Copy` is not satisfied
}

fn main() {}
