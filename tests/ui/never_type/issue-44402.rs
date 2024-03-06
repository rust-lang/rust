//@ check-pass

#![allow(dead_code)]
#![feature(never_type)]
#![feature(exhaustive_patterns)]

// Regression test for inhabitedness check. The old
// cache used to cause us to incorrectly decide
// that `test_b` was invalid.

struct Foo {
    field1: !,
    field2: Option<&'static Bar>,
}

struct Bar {
    field1: &'static Foo
}

fn test_a() {
    let x: Option<Foo> = None;
    match x { None => () }
}

fn test_b() {
    let x: Option<Bar> = None;
    match x {
        Some(_) => (),
        None => ()
    }
}

fn main() { }
