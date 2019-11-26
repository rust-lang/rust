// Test for <https://github.com/rust-lang/rust/issues/66756>

// check-pass

#![feature(const_if_match)]

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

fn main() {}
