//! Test for issue #52560 - derive Debug should show improved diagnostics
//! when type parameters are incorrectly assumed to need Debug bound

use std::fmt::Debug;

#[derive(Debug)]
struct Foo<B: Bar>(B::Item);

trait Bar {
    type Item: Debug;
}

fn foo<B: Bar>(f: Foo<B>) {
    println!("{:?}", f); //~ ERROR `B` doesn't implement `Debug`
}

struct ABC();

impl Bar for ABC {
    type Item = String;
}

fn main() {
    foo(Foo::<ABC>("a".into()));
}
