#![allow(unused)]
#![warn(clippy::large_const_arrays, clippy::large_stack_arrays)]

const ABOVE: [u8; 11] = [0; 11];
const BELOW: [u8; 10] = [0; 10];

fn main() {
    let above = [0u8; 11];
    let below = [0u8; 10];
}
