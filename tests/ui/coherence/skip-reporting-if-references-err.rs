// Regression test for #121006.
trait ToUnit<'a> {
    type Unit;
}

impl<T> ToUnit for T {}
//~^ ERROR implicit elided lifetime not allowed here

trait Overlap {}
impl<U> Overlap for fn(U) {}
impl Overlap for for<'a> fn(<() as ToUnit<'a>>::Unit) {}

fn main() {}
