//@ run-pass

struct Foo;

impl Foo {
    fn first() {}
}
impl Foo {
    fn second() {}
}

pub fn main() {
    Foo::first();
    Foo::second();
}
