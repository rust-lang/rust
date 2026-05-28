#![feature(impl_trait_in_bindings)]

trait Trait {}
impl<T: ?Sized> Trait for T {}

fn doesnt_work() {
    let x: &impl Trait = "hi";
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
}

fn works() {
    let x: &(impl Trait + ?Sized) = "hi";
    // No implicit sized.

    let x: &impl Trait = &();
    // Is actually sized.
}

fn main() {}
