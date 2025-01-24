// regression test for issue #121649.

trait ToUnit<'a> {
    type Unit;
}

trait Overlap<T> {}

type Assoc<'a, T> = <T as ToUnit<'a>>::Unit;

impl<T> Overlap<T> for T {}

impl<T> Overlap<for<'a> fn(&'a (), Assoc<'a, T>)> for T {}
//~^ ERROR the trait bound `for<'a> T: ToUnit<'a>` is not satisfied

fn main() {}
