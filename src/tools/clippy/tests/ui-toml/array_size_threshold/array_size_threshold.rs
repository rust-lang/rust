#![allow(unused)]
#![warn(clippy::large_const_arrays, clippy::large_stack_arrays)]
//@no-rustfix
const ABOVE: [u8; 11] = [0; 11];
//~^ large_const_arrays
const BELOW: [u8; 10] = [0; 10];

fn main() {
    let above = [0u8; 11];
    //~^ large_stack_arrays
    let below = [0u8; 10];
}
