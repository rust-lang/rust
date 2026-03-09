// Tests that namespaced crate names are limited to two segments

//@ aux-crate: std::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

use std::utils::get_u32;
//~^ ERROR unresolved import `std::utils`

fn main() {
    let _ = get_u32();
}
