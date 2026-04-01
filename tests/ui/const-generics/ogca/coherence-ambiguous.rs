//@ check-fail

#![feature(generic_const_items, min_generic_const_args, opaque_generic_const_args)]
#![expect(incomplete_features)]

type const FOO<const N: usize>: usize = const { N + 1 };

type const BAR<const N: usize>: usize = const { N + 1 };

trait Trait {}

impl Trait for [(); FOO::<1>] {}
impl Trait for [(); BAR::<1>] {}
//~^ ERROR conflicting implementations of trait `Trait` for type `[(); FOO::<1>]`
impl Trait for [(); BAR::<2>] {}
//~^ ERROR conflicting implementations of trait `Trait` for type `[(); FOO::<1>]`

fn main() {}
