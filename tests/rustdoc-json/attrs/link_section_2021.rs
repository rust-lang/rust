//@ edition: 2021
#![no_std]

//@ eq .index[] | select(.name == "example").attrs | ., ["#[link_section = \".text\"]"]
#[link_section = ".text"]
pub extern "C" fn example() {}
