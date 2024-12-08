//@ check-pass
#![feature(coerce_unsized, unsize)]

use std::marker::Unsize;
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized>(Box<T>);

impl<T> CoerceUnsized<Foo<dyn Baz>> for Foo<T> where T: Unsize<dyn Baz> {}

struct Bar;

trait Baz {}

impl Baz for Bar {}

fn main() {
    let foo = Foo(Box::new(Bar));
    let foobar: Foo<Bar> = foo;
}
