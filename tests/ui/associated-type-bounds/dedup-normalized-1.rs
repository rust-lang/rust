//@ check-pass

// We try to prove `T::Rigid: Into<?0>` and have 2 candidates from where-clauses:
//
// - `Into<String>`
// - `Into<<T::Rigid as Elaborate>::Assoc>`
//
// This causes ambiguity unless we normalize the alias in the second candidate
// to detect that they actually result in the same constraints.
trait Trait {
    type Rigid: Elaborate<Assoc = String> + Into<String>;
}

trait Elaborate: Into<Self::Assoc> {
    type Assoc;
}

fn impls<T: Into<U>, U>(_: T) {}

fn test<P: Trait>(rigid: P::Rigid) {
    impls(rigid);
}

fn main() {}
