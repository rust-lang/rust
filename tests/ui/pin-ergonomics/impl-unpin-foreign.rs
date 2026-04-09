//@ check-pass
#![feature(extern_types, pin_ergonomics)]
#![allow(incomplete_features)]

unsafe extern "C" {
    type ExternType;
}

impl Unpin for ExternType {}

fn main() {}
