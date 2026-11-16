#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(trait_alias)]

trait Bar<const N: usize> {}

trait BB = Bar<{ 2 + 1 }>;

fn foo(x: &dyn BB) {}
//~^ ERROR the trait alias `BB` is not dyn compatible [E0038]

fn main() {}
