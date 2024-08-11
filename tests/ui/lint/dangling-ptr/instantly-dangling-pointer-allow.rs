//@ check-pass

// This should not ICE.

#![allow(instantly_dangling_pointer)]

fn main() {
    dbg!(String::new().as_ptr());
}
