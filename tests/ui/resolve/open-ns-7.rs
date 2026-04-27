// Tests that namespaced crates cannot be resolved if shadowed.

//@ aux-crate: my_api=open-ns-my_api.rs
//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

use my_api::utils::utils_helper;
//~^ ERROR unresolved import `my_api::utils` [E0432]

fn main() {
    let _ = my_api::utils::utils_helper();
}
