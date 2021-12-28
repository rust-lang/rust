#![feature(generic_associated_types)]
#![allow(unused)]

trait Trait<'a> {
    type Foo;

    type Bar<'b>
    //~^ NOTE associated type defined here, with 1 lifetime parameter
    //~| NOTE
    where
        Self: 'b;
}

struct Impl<'a>(&'a ());

impl<'a> Trait<'a> for Impl<'a> {
    type Foo = &'a ();
    type Bar<'b> = &'b ();
}

type A<'a> = Impl<'a>;

type B<'a> = <A<'a> as Trait>::Foo;
//~^ ERROR missing lifetime specifier
//~| NOTE expected named lifetime parameter

type C<'a, 'b> = <A<'a> as Trait>::Bar;
//~^ ERROR missing lifetime specifier
//~| ERROR missing generics for associated type
//~| NOTE expected named lifetime parameter
//~| NOTE these named lifetimes are available to use
//~| NOTE expected 1 lifetime argument

fn main() {}
