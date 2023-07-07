// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]

struct Wrapper<G: Send>(G);

trait Foo {
    fn bar() -> Wrapper<impl Sized>;
    //~^ ERROR: cannot check whether the hidden type of opaque type satisfies auto traits
}

fn main() {}
