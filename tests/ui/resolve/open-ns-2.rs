//@ build-pass
//@ aux-crate: my_api=open-ns-my_api.rs
//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ aux-crate: my_api::core=open-ns-my_api_core.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

use my_api::core::{core_fn, core_fn2};
use my_api::utils::*;
use my_api::*;

fn main() {
    let _ = root_function();
    let _ = utils_helper();
    let _ = core_fn();
    let _ = core_fn2();
}
