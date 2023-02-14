#![feature(associated_type_bounds)]

trait Foo {
    type Assoc;
}

type X = dyn Foo<Assoc: Send>;
//~^ ERROR associated type bounds are unstable in this position
//~| ERROR unconstrained opaque type

fn main() {}
