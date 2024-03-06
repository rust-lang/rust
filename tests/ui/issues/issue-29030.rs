//@ check-pass
#![allow(dead_code)]
#[derive(Debug)]
struct Message<'a, P: 'a = &'a [u8]> {
    header: &'a [u8],
    payload: P,
}

fn main() {}
