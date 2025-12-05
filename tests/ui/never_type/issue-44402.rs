// Regression test for <https://github.com/rust-lang/rust/issues/44402>
//
// Previously inhabitedness check was handling cycles incorrectly causing this
// to not compile.
//
//@ check-pass

#![allow(dead_code)]
#![feature(never_type)]
#![feature(exhaustive_patterns)]

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

fn main() {}
