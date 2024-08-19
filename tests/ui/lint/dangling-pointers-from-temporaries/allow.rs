//@ check-pass

// This should not ICE.

#![allow(dangling_pointers_from_temporaries)]

fn main() {
    dbg!(String::new().as_ptr());
}
