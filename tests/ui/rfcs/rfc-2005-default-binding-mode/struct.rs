//@ run-pass
#[derive(Debug, PartialEq)]
struct Foo {
    x: u8,
}

pub fn main() {
    let mut foo = Foo {
        x: 1,
    };

    match &mut foo {
        Foo{x: n} => {
            *n += 1;
        },
    };

    assert_eq!(foo, Foo{x: 2});

    let Foo{x: n} = &foo;
    assert_eq!(*n, 2);
}
