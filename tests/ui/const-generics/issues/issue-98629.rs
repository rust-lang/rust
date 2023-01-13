#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    const N: usize;
}

impl const Trait for i32 {}
//~^ ERROR not all trait items implemented, missing: `N`

fn f()
where
    [(); <i32 as Trait>::N]:,
{}

fn main() {}
