//@ run-pass

#[derive(Debug)]
struct Foo(#[allow(dead_code)] Box<[u8]>);

pub fn main() {
    println!("{:?}", Foo(Box::new([0, 1, 2])));
}
