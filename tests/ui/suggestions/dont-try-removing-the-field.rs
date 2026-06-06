//@ check-pass

#![allow(dead_code)]

struct Foo {
    foo: i32,
    bar: i32,
    baz: (),
}

fn use_foo(x: Foo) -> (i32, i32) {
    let Foo { foo, bar, baz } = x;
    return (foo, bar);
}

fn main() {}
