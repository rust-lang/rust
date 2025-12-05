#![feature(portable_simd)]
use std::simd::Mask;

fn main() {
    let y = Mask::<_, _>::splat(false);
    //~^ ERROR: type annotations needed
    //~| ERROR type annotations needed
}
