#![feature(rustc_attrs)]
#![allow(internal_features)]

#[derive(Debug)]
#[rustc_layout_scalar_valid_range_start(2)]
struct NonZeroAndOneU8(u8);

fn main() {
    println!("{:?}", Some(1).map(NonZeroAndOneU8).unwrap());
    //~^ ERROR found `unsafe fn(u8) -> NonZeroAndOneU8 {NonZeroAndOneU8}`
}
