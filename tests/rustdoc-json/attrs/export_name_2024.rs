//@ edition: 2024
#![no_std]

// The representation of `#[unsafe(export_name = ..)]` in rustdoc in edition 2024
// is still `#[export_name = ..]` without the `unsafe` attribute wrapper.

//@ is "$.index[?(@.name=='example')].attrs" '["#[export_name = \"altered\"]"]'
#[unsafe(export_name = "altered")]
pub extern "C" fn example() {}
