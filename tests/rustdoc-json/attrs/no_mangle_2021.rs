//@ edition: 2021
#![no_std]

//@ jq .index[] | select(.name == "example").attrs == ["#[no_mangle]"]
#[no_mangle]
pub extern "C" fn example() {}
