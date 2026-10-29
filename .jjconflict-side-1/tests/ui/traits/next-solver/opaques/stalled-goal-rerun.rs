//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#267. This used to hang
// because a fast-path goal was not rerun after the opaque type storage changed.

trait Distribution<T> {}

impl Distribution<()> for u32 {}

impl<A, B> Distribution<(A, B)> for u32
where
    u32: Distribution<A>,
    u32: Distribution<B>,
{
}

trait Trait {
    type Item;
}

impl<T> Trait for Option<T>
where
    u32: Distribution<T>,
{
    type Item = T;
}

fn random_paulis() -> impl Trait<Item = ()> {
    None
}

fn main() {}
