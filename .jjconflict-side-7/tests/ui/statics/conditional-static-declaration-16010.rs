//! Regression test for https://github.com/rust-lang/rust/issues/16010

//@ run-pass
#![allow(dead_code)]

fn main() {
    if true { return }
    match () {
        () => { static MAGIC: usize = 0; }
    }
}
