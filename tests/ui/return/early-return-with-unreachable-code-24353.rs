//@ run-pass
#![allow(unreachable_code)]
fn main() {
    return ();

    let x = ();
    x
}

// https://github.com/rust-lang/rust/issues/24353
