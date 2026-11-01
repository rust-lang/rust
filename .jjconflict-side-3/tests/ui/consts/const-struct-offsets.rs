//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

enum Foo {
    IntVal(i32),
    Int64Val(i64)
}

struct Bar {
    i: i32,
    v: Foo
}

static bar: Bar = Bar { i: 0, v: Foo::IntVal(0) };

pub fn main() {}
