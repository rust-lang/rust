#![feature(rustc_attrs)]
#![allow(internal_features)]

#[derive(Debug)]
#[rustc_layout_scalar_valid_range_start(2)]
struct NonZeroAndOneU8(u8);

fn main() {
    println!("{:?}", Some(1).map(NonZeroAndOneU8).unwrap());
    //~^ ERROR expected a `FnOnce({integer})` closure, found `{fn item NonZeroAndOneU8: unsafe fn(u8) -> NonZeroAndOneU8}`
}
