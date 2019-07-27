// run-pass
// Regression test for issue #31267


struct Foo;

impl Foo {
    const FOO: [i32; 3] = [0; 3];
}

pub fn main() {
    let foo = Foo::FOO;
    assert_eq!(foo, [0i32, 0, 0]);
}
