#![feature(associated_type_defaults)]

// FIXME(#44265) add tests for type-generic and const-genertic associated types.

trait Foo {
    type A<'a>;
    type B<'a, 'b>;
    type C;
}

struct Fooy;

impl Foo for Fooy {
    type A = u32;
    //~^ ERROR lifetime parameters or bounds on type `A` do not match the trait declaration
    type B<'a, T> = Vec<T>;
    //~^ ERROR type `B` has 1 type parameter but its trait declaration has 0 type parameters
    type C<'a> = u32;
    //~^ ERROR lifetime parameters or bounds on type `C` do not match the trait declaration
}

struct Fooer;

impl Foo for Fooer {
    type A<T> = u32;
    //~^ ERROR type `A` has 1 type parameter but its trait declaration has 0 type parameters
    type B<'a> = u32;
    //~^ ERROR lifetime parameters or bounds on type `B` do not match the trait declaration
    type C<T> = T;
    //~^ ERROR type `C` has 1 type parameter but its trait declaration has 0 type parameters
}

fn main() {}
