//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ build-pass

//@ ignore-test FIXME(packages_as_namespaces): feature not implemented yet

use my_api; // FIXME can be resolved even though it (maybe?) shouldn't be
use my_api::utils::utils_helper;

fn main() {
    let _ = utils_helper();
}
