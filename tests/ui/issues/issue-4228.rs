// run-pass
// pretty-expanded FIXME #23616

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
