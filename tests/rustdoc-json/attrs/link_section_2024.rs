//@ edition: 2024
#![no_std]

// Since the 2024 edition the link_section attribute must use the unsafe qualification.
// However, the unsafe qualification is not shown by rustdoc.

//@ is "$.index[?(@.name=='example')].attrs" '["#[link_section = \".text\"]"]'
#[unsafe(link_section = ".text")]
pub extern "C" fn example() {}
