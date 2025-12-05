//@ run-pass
#[allow(dead_code)]
#[derive(Debug)]
struct Foo(isize, isize);

pub fn main() {
    let x = Foo(1, 2);
    println!("{:?}", x);
}
