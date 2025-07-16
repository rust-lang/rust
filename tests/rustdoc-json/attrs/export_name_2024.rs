//@ edition: 2024
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '["#[unsafe(export_name = \"altered\")]"]'
#[unsafe(export_name = "altered")]
pub extern "C" fn example() {}
