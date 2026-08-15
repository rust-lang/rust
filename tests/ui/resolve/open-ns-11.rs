// Tests that std has higher precedence than an open module with the same name.

//@ aux-crate: std::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

use std::utils::get_u32;
//~^ ERROR unresolved import `std::utils`

fn main() {
    let _ = get_u32();
}
