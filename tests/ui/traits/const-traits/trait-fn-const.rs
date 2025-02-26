// Regression test for issue #113378.
#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    const fn fun(); //~ ERROR functions in traits cannot be declared const
}

impl const Trait for () {
    const fn fun() {} //~ ERROR functions in trait impls cannot be declared const
}

impl Trait for u32 {
    const fn fun() {} //~ ERROR functions in trait impls cannot be declared const
}

trait NonConst {
    const fn fun(); //~ ERROR functions in traits cannot be declared const
}

fn main() {}
