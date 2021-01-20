#![feature(unboxed_closures)]

trait Trait {}

fn f<F:Trait(isize) -> isize>(x: F) {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
//~| ERROR E0220

fn main() {}
