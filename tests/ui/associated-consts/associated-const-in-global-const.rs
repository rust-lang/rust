//@ run-pass

struct Foo;

impl Foo {
    const BAR: f32 = 1.5;
}

const FOOBAR: f32 = <Foo>::BAR;

fn main() {
    assert_eq!(1.5f32, FOOBAR);
}
