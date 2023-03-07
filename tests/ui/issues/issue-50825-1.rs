// run-pass
// regression test for issue #50825
// Make sure that the `impl` bound (): X<T = ()> is preferred over
// the (): X bound in the where clause.

trait X {
    type T;
}

trait Y<U>: X {
    fn foo(x: &Self::T);
}

impl X for () {
    type T = ();
}

impl<T> Y<Vec<T>> for () where (): Y<T> {
    fn foo(_x: &()) {}
}

fn main () {}
