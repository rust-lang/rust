// Test for <https://github.com/rust-lang/rust/issues/66756>

//@ check-pass

enum E {
    A,
    B,
    C
}

const fn f(e: E) {
    match e {
        E::A => {}
        E::B => {}
        E::C => {}
    }
}

const fn g(e: E) -> usize {
    match e {
        _ => 0
    }
}

fn main() {
    const X: usize = g(E::C);
    assert_eq!(X, 0);
    assert_eq!(g(E::A), 0);
}
