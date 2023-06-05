#![feature(offset_of)]

use std::mem::offset_of;

enum Alpha {
    One(u8),
    Two(u8),
}

fn main() {
    offset_of!(Alpha::One, 0); //~ ERROR expected type, found variant `Alpha::One`
    offset_of!(Alpha, Two.0); //~ ERROR no field `Two` on type `Alpha`
}
