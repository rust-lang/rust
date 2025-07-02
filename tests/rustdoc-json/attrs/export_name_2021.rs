//@ edition: 2021
#![no_std]

//@ jq .index[] | select(.name == "example").attrs == ["#[export_name = \"altered\"]"]
#[export_name = "altered"]
pub extern "C" fn example() {}
