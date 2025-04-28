//@ edition: 2021
#![no_std]

//@ is "$.index[?(@.name=='example')].attrs" '[{"content": "#[export_name = \"altered\"]", "is_inner": false}]'
#[export_name = "altered"]
pub extern "C" fn example() {}
