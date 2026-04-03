// Regression test for #138891
// This used to ICE with "encountered a fresh type during canonicalization"
// when using a trait alias with `Self` in a dyn context with extra generic args.

#![feature(trait_alias)]

trait F = Fn() -> Self;

fn _f3<Fut>(a: dyn F<Fut>) {}
//~^ ERROR trait alias takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR associated type binding in trait object type mentions `Self`

fn main() {}
