// Regression test for issue #155834

#![expect(incomplete_features)]
#![feature(min_generic_const_args, macroless_generic_const_args)]

trait Trait {}

impl<'t> Trait for [(); N] {}
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for implementations

fn N(arg: impl Trait) {}

fn main() {}
