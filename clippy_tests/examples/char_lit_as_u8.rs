#![feature(plugin)]
#![plugin(clippy)]

#![warn(char_lit_as_u8)]
#![allow(unused_variables)]
fn main() {
    let c = 'a' as u8;
}
