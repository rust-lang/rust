//@ check-pass

// This used to ICE, because the compiler confused a pointer-like to dyn* coercion
// with a c-like enum to integer cast.

#![feature(dyn_star, pointer_like_trait)]
#![expect(incomplete_features)]

use std::marker::PointerLike;

#[repr(transparent)]
enum E {
    Num(usize),
}

impl PointerLike for E {}

trait Trait {}
impl Trait for E {}

fn main() {
    let _ = E::Num(42) as dyn* Trait;
}
