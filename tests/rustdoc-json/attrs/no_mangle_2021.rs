//@ edition: 2021
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["#[no_mangle]\n"]'
#[no_mangle]
pub extern "C" fn example() {}
