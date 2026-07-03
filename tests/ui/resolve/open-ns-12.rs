//@ build-pass
//@ aux-crate: my_api=open-ns-my_api.rs
//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

extern crate my_api;

use my_api::utils::utils_helper;

fn main() {
    let _ = utils_helper();
    let _ = my_api::utils::utils_helper();
}
