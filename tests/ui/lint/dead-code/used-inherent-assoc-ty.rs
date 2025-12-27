//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#[deny(dead_code)]

fn main() {
    let _: Struct::Item = ();
}

struct Struct;
impl Struct { type Item = (); }
