//@ run-pass
#![allow(dead_code)]
enum Foo {
    Bar(Option<i8>, (), (), Vec<i32>),
    Baz,
}

pub fn main() {
    let foo = Foo::Bar(Some(1), (), (), vec![2, 3]);

    match &foo {
        Foo::Baz => panic!(),
        Foo::Bar(None, ..) => panic!(),
        Foo::Bar(Some(n), .., v) => {
            assert_eq!((*v).len(), 2);
            assert_eq!(*n, 1);
        }
    }
}
