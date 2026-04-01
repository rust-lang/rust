#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

type Sign = pattern_type!(u32 is -10..);
//~^ ERROR: cannot apply unary operator `-`

type SignedChar = pattern_type!(char is -'A'..);
//~^ ERROR: cannot apply unary operator `-`

fn main() {
    match 42_u8 {
        -10..253 => {}
        //~^ ERROR `u8: Neg` is not satisfied
        _ => {}
    }

    match 'A' {
        -'\0'..'a' => {}
        //~^ ERROR `char: Neg` is not satisfied
        _ => {}
    }
}
