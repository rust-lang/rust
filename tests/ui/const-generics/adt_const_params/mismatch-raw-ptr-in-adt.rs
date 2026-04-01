//! Regression test for <https://github.com/rust-lang/rust/issues/131052>

#![feature(adt_const_params)]

struct ConstBytes<const T: &'static [*mut u8; 3]>;
//~^ ERROR `&'static [*mut u8; 3]` can't be used as a const parameter type

pub fn main() {
    let _: ConstBytes<b"AAA"> = ConstBytes::<b"BBB">;
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}
