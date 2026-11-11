//! regression test for <https://github.com/rust-lang/rust/issues/31267>
//@ run-pass


struct Foo;

impl Foo {
    const FOO: [i32; 3] = [0; 3];
}

pub fn main() {
    let foo = Foo::FOO;
    assert_eq!(foo, [0i32, 0, 0]);
}
