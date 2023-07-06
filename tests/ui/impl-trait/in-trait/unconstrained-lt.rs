// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]

trait Foo {
    fn test() -> impl Sized;
}

impl<'a, T> Foo for T {
    fn test() -> &'a () { &() }
    //~^ ERROR return type captures more lifetimes than trait definition
}

fn main() {}
