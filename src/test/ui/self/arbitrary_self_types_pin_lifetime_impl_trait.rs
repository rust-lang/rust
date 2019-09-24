// compile-fail

use std::pin::Pin;

struct Foo;

impl Foo {
    fn f(self: Pin<&Self>) -> impl Clone { self } //~ ERROR cannot infer an appropriate lifetime
}

fn main() {
    { Pin::new(&Foo).f() };
}
