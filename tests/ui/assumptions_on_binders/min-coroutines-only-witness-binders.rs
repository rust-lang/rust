//@ compile-flags: -Zassumptions-on-binders=min_coroutines

use std::marker::PhantomData;

struct WellFormed<'a, T: 'a>(PhantomData<&'a T>);

trait Trait {}

impl<'a, 'b> Trait for WellFormed<'a, &'b ()>
where
    &'b (): 'a,
{
}

fn require()
where
    for<'a, 'b> WellFormed<'a, &'b ()>: Trait,
{
}

fn check() {
    require();
    //~^ ERROR type annotations needed: cannot satisfy
}

fn main() {}
