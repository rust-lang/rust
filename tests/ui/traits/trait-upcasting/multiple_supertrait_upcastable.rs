#![feature(multiple_supertrait_upcastable)]
#![deny(multiple_supertrait_upcastable)]

trait A {}
trait B {}

trait C: A + B {}
//~^ ERROR `C` is dyn-compatible and has multiple supertraits

fn main() {}
