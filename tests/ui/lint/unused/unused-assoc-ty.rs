#![deny(unused_must_use)]

trait Foo {
    #[must_use]
    type Result;

    fn process(arg: i32) -> Self::Result;
}

fn generic<T: Foo>() {
    T::process(10);
    //~^ ERROR unused `Foo::Result` that must be used
}

struct NoOp;
impl Foo for NoOp {
    type Result = ();

    fn process(_arg: i32) { println!("did nothing"); }
}

fn noop() {
    // Should not lint.
    <NoOp as Foo>::process(10);
}

fn main() {}
