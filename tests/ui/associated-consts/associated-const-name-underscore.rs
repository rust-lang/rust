//@ run-pass
#![feature(associated_const_underscore)]
struct Thing {}

impl Thing {
    const _: () = {};
    const _: u8 = 0;
}

fn main() {
    let _ = Thing {};
}
