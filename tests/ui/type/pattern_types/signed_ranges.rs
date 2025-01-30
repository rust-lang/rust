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
        _ => {}
    }

    match 'A' {
        -'\0'..'a' => {}
        _ => {}
    }
}
