//@ edition: 2021
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '[{"content": "#[no_mangle]", "is_inner": false}]'
#[no_mangle]
pub extern "C" fn example() {}
