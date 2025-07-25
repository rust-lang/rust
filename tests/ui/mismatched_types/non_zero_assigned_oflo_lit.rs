#![allow(overflowing_literals)]

fn main() {
    let _: std::num::NonZero<u8> = 256;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`

    let _: std::num::NonZero<i8> = -128;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
    let _: std::num::NonZero<i8> = -129;
    //~^ ERROR mismatched types
    //~| HELP  consider calling `NonZero::new`
}
