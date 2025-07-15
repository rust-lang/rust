//@ edition: 2021
#![no_std]

//@ count "$.index[?(@.name=='example')].attrs[*]" 1
//@ is "$.index[?(@.name=='example')].attrs[*].link_section" '".text"'
#[link_section = ".text"]
pub extern "C" fn example() {}
