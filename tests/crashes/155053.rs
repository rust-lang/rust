//@ known-bug: #155053
#![feature(pin_ergonomics)]
#![feature(extern_types)]

unsafe extern "C" {
    type ExternType;
}

impl Unpin for ExternType {}

fn main() {}
