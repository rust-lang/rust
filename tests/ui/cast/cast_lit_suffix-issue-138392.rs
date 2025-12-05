//@ run-rustfix
#![allow(unused_parens)]
fn main() {
    let _x: u8 = (4i32); //~ ERROR: mismatched types
    let _y: u8 = (4.0f32); //~ ERROR: mismatched types
}
