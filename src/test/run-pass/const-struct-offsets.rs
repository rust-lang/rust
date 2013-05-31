enum Foo {
    IntVal(i32),
    Int64Val(i64)
}

struct Bar {
    i: i32,
    v: Foo
}

static bar: Bar = Bar { i: 0, v: IntVal(0) };

fn main() {}

