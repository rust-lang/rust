#![deny(range_syntax)]
#![allow(dead_code)]

fn main() {
    0..1; //~ ERROR usage of `Range` range syntax
    
    2..=3; //~ ERROR usage of `RangeInclusive` range syntax

    4..; //~ ERROR usage of `RangeFrom` range syntax

    // Should no trigger for others.
    ..5;
    ..=6;
    ..;
}
