//! Regression test for https://github.com/rust-lang/rust/issues/4387

//@ run-pass

pub fn main() {
    let _foo = [0; 2*4];
}
