#![feature(const_trait_impl)]

trait Trait {
    const N: usize;
}

// FIXME: We should mention that `N` is missing
impl const Trait for i32 {}

fn f()
where
    [(); <i32 as Trait>::N]:,
    //~^ ERROR unable to use constant with a hidden value in the type system
{}

fn main() {}
