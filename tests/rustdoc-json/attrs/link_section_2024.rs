//@ edition: 2024
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["#[unsafe(link_section = \".text\")]"]'
#[unsafe(link_section = ".text")]
pub extern "C" fn example() {}
