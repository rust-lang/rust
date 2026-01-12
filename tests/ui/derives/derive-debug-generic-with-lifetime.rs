//! regression test for <https://github.com/rust-lang/rust/issues/29030>
//@ check-pass
#![allow(dead_code)]
#[derive(Debug)]
struct Message<'a, P: 'a = &'a [u8]> {
    header: &'a [u8],
    payload: P,
}

fn main() {}
