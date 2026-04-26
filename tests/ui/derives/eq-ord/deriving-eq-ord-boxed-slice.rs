//@ run-pass
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug)]
struct Foo(Box<[u8]>);

pub fn main() {
    let a = Foo(Box::new([0, 1, 2]));
    let b = Foo(Box::new([0, 1, 2]));
    assert_eq!(a, b);
    println!("{}", a != b);
    println!("{}", a < b);
    println!("{}", a <= b);
    println!("{}", a == b);
    println!("{}", a > b);
    println!("{}", a >= b);
}
