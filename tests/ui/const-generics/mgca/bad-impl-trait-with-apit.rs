// Regression test for issue #155834

#![expect(incomplete_features)]
#![feature(min_generic_const_args, macroless_generic_const_args)]

trait Trait {}

impl<'t> Trait for [(); N] {}
//~^ ERROR function items cannot be used as const args

fn N(arg: impl Trait) {}

fn main() {}
