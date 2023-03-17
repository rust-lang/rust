// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
//~^ WARN the feature `return_position_impl_trait_in_trait` is incomplete

trait Trait {
    type Type;

    // Check that we're adjusting bound vars correctly when installing the default
    // method projection assumptions.
    fn method(&self) -> impl Trait<Type = impl Sized + '_>;
}

fn main() {}
