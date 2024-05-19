//@ run-pass
#![feature(box_patterns)]

struct Foo { a: isize, b: isize }

pub fn main() {
    let box Foo{ a, b } = Box::new(Foo { a: 100, b: 200 });
    assert_eq!(a + b, 300);
}
