//@ compile-flags: -Znext-solver
//@ check-pass

trait Id {
    type Assoc;
}
impl<T> Id for T {
    type Assoc = T;
}


// Coherence should be able to reason that `<LocalTy as Id>::Assoc: Copy`
// does not hold.
//
// See https://github.com/rust-lang/trait-system-refactor-initiative/issues/51
// for more details.
trait Trait {}
impl<T: Copy> Trait for T {}
struct LocalTy;
impl Trait for <LocalTy as Id>::Assoc {}

fn main() {}
