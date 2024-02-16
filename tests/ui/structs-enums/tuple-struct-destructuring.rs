//@ run-pass
struct Foo(isize, isize);

pub fn main() {
    let x = Foo(1, 2);
    let Foo(y, z) = x;
    println!("{} {}", y, z);
    assert_eq!(y, 1);
    assert_eq!(z, 2);
}
