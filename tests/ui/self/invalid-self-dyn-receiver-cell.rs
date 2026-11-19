// Regression test for rust-lang/rust#165033.
// An invalid receiver must not trigger an ICE during method confirmation.

use std::cell::Cell;
use std::ops::Deref;

trait Foo: Deref<Target = Cell<dyn Bar>> {
    fn method(self: &Cell<dyn Bar>) {}
    //~^ ERROR invalid `self` parameter type
}

trait Bar {}

fn test(x: &dyn Foo) {
    x.method();
}

fn main() {}
