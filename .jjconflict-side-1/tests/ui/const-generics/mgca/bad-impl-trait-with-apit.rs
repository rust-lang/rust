// Regression test for issue #155834

#![expect(incomplete_features)]
#![feature(gca_min_const_items, gca_macroless_args)]

trait Trait {}

impl<'t> Trait for [(); N] {}
//~^ ERROR function items cannot be used as const args

fn N(arg: impl Trait) {}

fn main() {}
