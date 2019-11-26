// check-pass

#![feature(const_if_match)]

enum E {
    A,
    B,
    C
}

const fn f(e: E) -> usize {
    match e {
        _ => 0
    }
}

fn main() {
    const X: usize = f(E::C);
    assert_eq!(X, 0);
    assert_eq!(f(E::A), 0);
}
