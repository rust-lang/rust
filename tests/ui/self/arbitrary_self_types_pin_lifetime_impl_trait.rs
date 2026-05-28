use std::pin::Pin;

struct Foo;

impl Foo {
    fn f(self: Pin<&Self>) -> impl Clone { self }
    //~^ ERROR: captures lifetime that does not appear in bounds
}

fn main() {
    { Pin::new(&Foo).f() };
}
