//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Ensure we don't have ambiguity when upcasting to two supertraits
// that are identical modulo normalization.

trait Supertrait<T> {
    fn method(&self) {}
}
impl<T> Supertrait<T> for () {}

trait Identity {
    type Selff;
}
impl<Selff> Identity for Selff {
    type Selff = Selff;
}
trait Trait<P>: Supertrait<()> + Supertrait<<P as Identity>::Selff> {}

impl<P> Trait<P> for () {}

fn main() {
    let x: &dyn Trait<()> = &();
    let x: &dyn Supertrait<()> = x;
}
