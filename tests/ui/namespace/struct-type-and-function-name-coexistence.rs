//@ run-pass

struct A {
    a: isize,
}

fn a(a: A) -> isize {
    return a.a;
}

pub fn main() {
    let x: A = A { a: 1 };
    assert_eq!(a(x), 1);
}
