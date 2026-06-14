//! regression test for https://github.com/rust-lang/rust/issues/18110
//@ check-pass
#![allow(unreachable_code)]

fn main() {
    ({ return },);
}
