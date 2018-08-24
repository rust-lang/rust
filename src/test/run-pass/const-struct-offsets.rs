// pretty-expanded FIXME #23616

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
