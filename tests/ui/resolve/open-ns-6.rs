// Tests that virtual modules are resolvable.

//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ check-pass

use my_api;
use my_api::utils::utils_helper;

fn main() {
    let _ = utils_helper();
}
