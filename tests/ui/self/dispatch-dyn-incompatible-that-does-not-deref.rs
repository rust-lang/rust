// Regression test for <https://github.com/rust-lang/rust/issues/141419>.

use std::ops::Deref;

struct W;

trait Foo: Deref<Target = W> {
    fn method(self: &W) {}
    //~^ ERROR invalid `self` parameter type: `&W`
}

fn test(x: &dyn Foo) {
    //~^ ERROR the trait `Foo` is not dyn compatible
    x.method();
    //~^ ERROR the trait `Foo` is not dyn compatible
}

fn main() {}
