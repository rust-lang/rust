#![deny(range_syntax)]
#![allow(dead_code)]

fn main() {
    0..1; //~ ERROR usage of range syntax
    
    2..=3; //~ ERROR usage of range syntax

    4..; //~ ERROR usage of range syntax

    ..5;

    ..=6;

    ..;
}
