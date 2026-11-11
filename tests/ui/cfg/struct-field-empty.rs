//! regression test for <https://github.com/rust-lang/rust/issues/16819>
//@ run-pass
#![allow(unused_variables)]
// `#[cfg]` on struct field permits empty unusable struct

struct S {
    #[cfg(false)]
    a: int,
}

fn main() {
    let s = S {};
}
