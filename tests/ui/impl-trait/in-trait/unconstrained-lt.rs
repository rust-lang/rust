#![feature(return_position_impl_trait_in_trait)]

trait Foo {
    fn test() -> impl Sized;
}

impl<'a, T> Foo for T {
    //~^ ERROR the lifetime parameter `'a` is not constrained by the impl trait, self type, or predicates

    fn test() -> &'a () {
        &()
    }
}

fn main() {}
