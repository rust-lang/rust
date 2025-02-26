//@ only-x86_64

#![feature(dyn_star, pointer_like_trait)]
//~^ WARN the feature `dyn_star` is incomplete

use std::fmt::Debug;
use std::marker::PointerLike;

fn make_dyn_star() -> dyn* Debug + 'static {
    f32::from_bits(0x1) as f64
    //~^ ERROR `f64` needs to have the same ABI as a pointer
}

fn main() {
    println!("{:?}", make_dyn_star());
}
