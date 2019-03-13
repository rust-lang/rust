#![allow(unused_must_use)]

use std::convert::TryFrom;
use std::num::TryFromIntError;

fn main() {
    let x: u32 = 125;
    let y: Result<u8, TryFromIntError> = u8::try_from(x);
    y == Ok(125);
}
