#![feature(generic_associated_types)]
//~^ WARN the feature `generic_associated_types` is incomplete

fn main() {}

struct X;

impl X {
    type Y;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR associated types are not yet supported in inherent impls
    type Z: Ord;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR bounds on associated `type`s in `impl`s have no effect
    //~| ERROR associated types are not yet supported in inherent impls
    type W: Ord where Self: Eq;
    //~^ ERROR associated type in `impl` without body
    //~| ERROR bounds on associated `type`s in `impl`s have no effect
    //~| ERROR associated types are not yet supported in inherent impls
    type W where Self: Eq;
    //~^ ERROR associated type in `impl` without body
}
