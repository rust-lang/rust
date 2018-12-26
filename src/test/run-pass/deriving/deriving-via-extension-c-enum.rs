// run-pass
#![allow(dead_code)]
#[derive(PartialEq, Debug)]
enum Foo {
    Bar,
    Baz,
    Boo
}

pub fn main() {
    let a = Foo::Bar;
    let b = Foo::Bar;
    assert_eq!(a, b);
    assert!(!(a != b));
    assert!(a.eq(&b));
    assert!(!a.ne(&b));
}
