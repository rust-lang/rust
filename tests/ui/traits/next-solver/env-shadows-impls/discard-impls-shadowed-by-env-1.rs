//@ compile-flags: -Znext-solver
//@ check-pass

// Normalizing `<T as Trait>::TraitAssoc` in the elaborated environment
// `[T: Trait, T: Super, <T as Super>::SuperAssoc = <T as Trait>::TraitAssoc]`
// has a single impl candidate, which uses the environment to
// normalize `<T as Trait>::TraitAssoc` to itself. We avoid this overflow
// by discarding impl candidates the trait bound is proven by a where-clause.

// https://github.com/rust-lang/trait-system-refactor-initiative/issues/76
trait Super {
    type SuperAssoc;
}

trait Trait: Super<SuperAssoc = Self::TraitAssoc> {
    type TraitAssoc;
}

impl<T, U> Trait for T
where
    T: Super<SuperAssoc = U>,
{
    type TraitAssoc = U;
}

fn overflow<T: Trait>() {
    let x: <T as Trait>::TraitAssoc;
}

fn main() {}
