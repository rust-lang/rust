// FIXME(#84561): function-like macros produce unintuitive coverage results.
// This test demonstrates some of the problems.

#[derive(Debug, PartialEq, Eq)]
struct Foo(u32);

fn main() {
    let bar = Foo(1);
    assert_eq!(bar, Foo(1));
    let baz = Foo(0);
    assert_ne!(baz, Foo(1));
    println!("{:?}", Foo(1));
    println!("{:?}", bar);
    println!("{:?}", baz);

    assert_eq!(Foo(1), Foo(1));
    assert_ne!(Foo(0), Foo(1));
    assert_eq!(Foo(2), Foo(2));
    let bar = Foo(1);
    assert_ne!(Foo(0), Foo(3));
    assert_ne!(Foo(0), Foo(4));
    assert_eq!(Foo(3), Foo(3));
    assert_ne!(Foo(0), Foo(5));
    println!("{:?}", bar);
    println!("{:?}", Foo(1));
}
