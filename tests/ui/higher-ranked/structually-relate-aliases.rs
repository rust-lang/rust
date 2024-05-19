// regression test for issue #121649.

trait ToUnit<'a> {
    type Unit;
}

trait Overlap<T> {}

type Assoc<'a, T> = <T as ToUnit<'a>>::Unit;

impl<T> Overlap<T> for T {}

impl<T> Overlap<for<'a> fn(&'a (), Assoc<'a, T>)> for T {}
//~^ ERROR 13:17: 13:49: the trait bound `for<'a> T: ToUnit<'a>` is not satisfied [E0277]
//~| ERROR 13:36: 13:48: the trait bound `for<'a> T: ToUnit<'a>` is not satisfied [E0277]

fn main() {}
