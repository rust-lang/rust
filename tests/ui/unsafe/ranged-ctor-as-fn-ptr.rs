#![feature(rustc_attrs)]

#[derive(Debug)]
#[rustc_layout_scalar_valid_range_start(2)]
struct NonZeroAndOneU8(u8);

fn main() {
    let x: fn(u8) -> NonZeroAndOneU8 = NonZeroAndOneU8;
    //~^ ERROR mismatched types
}
