// run-pass

#![allow(dead_code)]

#[derive(Debug)]
struct Foo {
    x: isize,
    y: isize
}

pub fn main() {
    let a = Foo { x: 1, y: 2 };
    let c = Foo { x: 4, .. a};
    println!("{:?}", c);
}
