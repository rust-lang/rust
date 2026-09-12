// Shows that rust-lang/rust/83601 is resolved

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
}
