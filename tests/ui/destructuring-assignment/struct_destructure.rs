//@ run-pass

struct Struct<S, T> {
    a: S,
    b: T,
}

fn main() {
    let (mut a, mut b);
    Struct { a, b } = Struct { a: 0, b: 1 };
    assert_eq!((a, b), (0, 1));
    Struct { a: b, b: a }  = Struct { a: 1, b: 2 };
    assert_eq!((a,b), (2, 1));
    Struct { a: _, b } = Struct { a: 1, b: 2 };
    assert_eq!((a, b), (2, 2));
    Struct { a, .. } = Struct { a: 1, b: 3 };
    assert_eq!((a, b), (1, 2));
    Struct { .. } = Struct { a: 1, b: 4 };
    assert_eq!((a, b), (1, 2));
}
