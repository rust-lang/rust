// Regression test for issue #52560. An unsatisfied bound introduced by an
// imperfect derive should be explained where the derived implementation is
// generated.

use std::fmt::Debug;

#[derive(Debug)]
struct Foo<B: Bar>(B::Item);

trait Bar {
    type Item: Debug;
}

fn foo<B: Bar>(value: Foo<B>) {
    println!("{value:?}");
    //~^ ERROR `B` doesn't implement `Debug`
}

struct Concrete;

impl Bar for Concrete {
    type Item = String;
}

fn main() {
    foo(Foo::<Concrete>("value".into()));
}
