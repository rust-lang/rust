// Regression test for issue #113378.
#![feature(const_trait_impl, effects)]

#[const_trait]
trait Trait {
    const fn fun(); //~ ERROR functions in traits cannot be declared const
}

impl const Trait for () {
    const fn fun() {}  //~ ERROR functions in trait impls cannot be declared const
}

fn main() {}
