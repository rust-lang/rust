//! regression test for https://github.com/rust-lang/rust/issues/18110
//@ run-pass
#![allow(unreachable_code)]

fn main() {
    ({ return },);
}
