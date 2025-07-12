//@ edition: 2024
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["#[unsafe(no_mangle)]"]'
#[unsafe(no_mangle)]
pub extern "C" fn example() {}
