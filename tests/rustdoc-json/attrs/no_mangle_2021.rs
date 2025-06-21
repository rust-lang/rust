//@ edition: 2021
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["#[attr = NoMangle]"]'
#[no_mangle]
pub extern "C" fn example() {}
