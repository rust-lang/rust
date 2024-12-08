//@ run-pass
//@ compile-flags: -Z mir-opt-level=4

// Checks that the compiler does not ICE when passing references to field of by-value struct
// with -Z mir-opt-level=4

fn do_nothing(_: &()) {}

pub struct Foo {
    bar: (),
}

pub fn by_value_1(foo: Foo) {
    do_nothing(&foo.bar);
}

pub fn by_value_2<T>(foo: Foo) {
    do_nothing(&foo.bar);
}

fn main() {
    by_value_1(Foo { bar: () });
    by_value_2::<()>(Foo { bar: () });
}
