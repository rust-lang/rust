//! Ensure overflowing literals are not allowed for
//! layout constrained types like `NonZero`, as that makes
//! it harder to perform the layout checks. Instead
//! we hard error if such literals are out of range.

#![allow(overflowing_literals)]

fn main() {
    let _: std::num::NonZero<u8> = 256;
    //~^ ERROR literal out of range

    let _: std::num::NonZero<i8> = -128;
    let _: std::num::NonZero<i8> = -129;
    //~^ ERROR literal out of range
}
