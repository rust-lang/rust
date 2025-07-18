//@ edition: 2024
#![no_std]

// The representation of `#[unsafe(no_mangle)]` in rustdoc in edition 2024
// is still `#[no_mangle]` without the `unsafe` attribute wrapper.

//@ is "$.index[?(@.name=='example')].attrs" '["no_mangle"]'
#[unsafe(no_mangle)]
pub extern "C" fn example() {}
