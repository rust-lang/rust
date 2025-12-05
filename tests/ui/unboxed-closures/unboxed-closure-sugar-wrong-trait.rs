#![feature(unboxed_closures)]

trait Trait {}

fn f<F:Trait(isize) -> isize>(x: F) {}
//~^ ERROR trait takes 0 generic arguments but 1 generic argument
//~| ERROR associated type `Output` not found for `Trait`

fn main() {}
