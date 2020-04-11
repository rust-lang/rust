// run-pass

#![feature(destructuring_assignment)]
struct Struct<S, T> {
    a: S,
    b: T,
}

fn main() {
    let (a, b);
    Struct { a, b } = Struct { a: 0, b: 1 };
    assert_eq!((a, b), (0, 1));
}
