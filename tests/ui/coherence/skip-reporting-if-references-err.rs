// Regression test for #121006.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait ToUnit<'a> {
    type Unit;
}

impl<T> ToUnit for T {}
//~^ ERROR implicit elided lifetime not allowed here

trait Overlap {}
impl<U> Overlap for fn(U) {}
impl Overlap for for<'a> fn(<() as ToUnit<'a>>::Unit) {}
//[current]~^ ERROR the trait bound `for<'a> (): ToUnit<'a>` is not satisfied
//[current]~| ERROR the trait bound `for<'a> (): ToUnit<'a>` is not satisfied

fn main() {}
