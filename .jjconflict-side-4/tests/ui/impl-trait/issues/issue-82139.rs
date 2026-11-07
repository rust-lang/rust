#![feature(impl_trait_in_assoc_type)]

trait Trait {
    type Associated;
    fn func() -> Self::Associated;
}

trait Bound {}
pub struct Struct;

impl Trait for Struct {
    type Associated = impl Bound;

    fn func() -> Self::Associated {
        Some(42).map(|_| j) //~ ERROR cannot find value `j` in this scope
    }
}

fn main() {}
