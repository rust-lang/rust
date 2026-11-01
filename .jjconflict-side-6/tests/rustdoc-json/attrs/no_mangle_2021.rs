//@ edition: 2021
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["no_mangle"]'
#[no_mangle]
pub extern "C" fn example() {}
