//@ run-pass
#![allow(dead_code)]
use std::mem;

struct Foo<T: ?Sized> {
    a: i64,
    b: bool,
    c: T,
}

fn main() {
    let foo: &Foo<i32> = &Foo { a: 1, b: false, c: 2i32 };
    let foo_unsized: &Foo<dyn Send> = foo;
    assert_eq!(mem::size_of_val(foo), mem::size_of_val(foo_unsized));
}
