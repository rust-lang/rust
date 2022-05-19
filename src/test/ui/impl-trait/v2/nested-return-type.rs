// Check that nested impl Trait items work in functions with generic parameters.
// check-pass

#![feature(return_position_impl_trait_v2)]

fn nested_assoc_type<'a, T>() -> impl Iterator<Item = impl Sized> {
    [1].iter()
}

fn main() {}
