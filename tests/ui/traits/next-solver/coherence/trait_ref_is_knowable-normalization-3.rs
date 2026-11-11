//@ compile-flags: -Znext-solver
//@ check-pass

trait Id {
    type Assoc;
}
impl<T> Id for T {
    type Assoc = T;
}


// Coherence should be able to reason that `(): PartialEq<<LocalTy as Id>::Assoc>>`
// does not hold.
//
// See https://github.com/rust-lang/trait-system-refactor-initiative/issues/51
// for more details.
trait Trait {}
impl<T> Trait for T
where
    (): PartialEq<T> {}
struct LocalTy;
impl Trait for <LocalTy as Id>::Assoc {}

fn main() {}
