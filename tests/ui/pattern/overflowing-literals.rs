//! Check that overflowing literals are in patterns are rejected

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

// FIXME(pattern_types): also reject overflowing literals
type TooBig = pattern_type!(u8 is 500..);
type TooSmall = pattern_type!(i8 is -500..);

fn main() {
    match 5_u8 {
        500 => {}
        //~^ ERROR literal out of range for `u8`
        _ => {}
    }
}
