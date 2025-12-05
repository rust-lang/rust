//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#84.
//
// We try to infer `T::Rigid: Into<?0>` and have 2 candidates from where-clauses:
//
// - `Into<String>`
// - `Into<<T::Rigid as Elaborate>::Assoc>`
//
// This causes ambiguity unless we normalize the alias in the second candidate
// to detect that they actually result in the same constraints.
trait Trait {
    type Rigid: Elaborate<Assoc = String> + Into<String> + Default;
}

trait Elaborate: Into<Self::Assoc> {
    type Assoc;
}

fn test<T: Trait>() {
    let rigid: T::Rigid = Default::default();
    drop(rigid.into());
}

fn main() {}
