//@ run-pass
#![allow(dead_code)]
#[derive(Debug, PartialEq)]
enum Bar {
    A(i64),
    B(i32),
    C,
}

#[derive(Debug, PartialEq)]
struct Foo(Bar, u8);

static FOO: [Foo; 2] = [Foo(Bar::C, 0), Foo(Bar::C, 0xFF)];

fn main() {
    assert_eq!(&FOO[1],  &Foo(Bar::C, 0xFF));
}
