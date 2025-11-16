#![allow(overflowing_literals)]

fn main() {
    let x: std::num::NonZero<i8> = -128;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    assert_eq!(x.get(), -128_i8);
}
