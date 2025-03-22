//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// We try to prove `for<'b> T::Rigid: Bound<'b, ?0>` and have 2 candidates from where-clauses:
//
// - `for<'a> Bound<'a, String>`
// - `for<'a> Bound<'a, <T::Rigid as Elaborate>::Assoc>`
//
// This causes ambiguity unless we normalize the alias in the second candidate
// to detect that they actually result in the same constraints. We currently
// fail to detect that the constraints from these bounds are equal and error
// with ambiguity.
trait Bound<'a, U> {}

trait Trait {
    type Rigid: Elaborate<Assoc = String> + for<'a> Bound<'a, String>;
}

trait Elaborate: for<'a> Bound<'a, Self::Assoc> {
    type Assoc;
}

fn impls<T: for<'b> Bound<'b, U>, U>(_: T) {}

fn test<P: Trait>(rigid: P::Rigid) {
    impls(rigid);
    //[current]~^ ERROR type annotations needed
}

fn main() {}
