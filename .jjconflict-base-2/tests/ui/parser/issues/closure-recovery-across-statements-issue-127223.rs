//! Regression test for https://github.com/rust-lang/rust/issues/127223.
//! A semicolon in a later statement must not trigger recovery of an earlier closure.

fn direct_closure() {
    let value = || {};
    (value;);
    //~^ ERROR expected one of
}

fn closure_argument() {
    let value = std::iter::once_with(|| ());
    std::mem::drop(value;);
    //~^ ERROR expected one of
}

fn main() {}
