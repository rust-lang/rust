//@ known-bug: #113280
//@ only-x86_64

#![feature(dyn_star, pointer_like_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;
use std::marker::PointerLike;

fn make_dyn_star<'a>(t: impl PointerLike + Debug + 'a) -> dyn* Debug + 'a {
    f32::from_bits(0x1) as f64
}

fn main() {
    println!("{:?}", make_dyn_star(Box::new(1i32)));
}
