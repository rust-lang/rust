//@ run-pass

struct Foo;

impl Foo {
    const ID: i32 = 1;
}

fn main() {
    assert_eq!(1, Foo::ID);
}
