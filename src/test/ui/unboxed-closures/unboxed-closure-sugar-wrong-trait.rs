#![feature(unboxed_closures)]

trait Trait {}

fn f<F:Trait(isize) -> isize>(x: F) {}
//~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied
//~| ERROR associated type `Output` not found for `Trait`

fn main() {}
