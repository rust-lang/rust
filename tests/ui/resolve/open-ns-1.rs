//@ build-pass
//@ aux-crate:my_api=open-ns-my_api.rs
//@ aux-crate:my_api::utils=open-ns-my_api_utils.rs
//@ aux-crate:my_api::core=open-ns-my_api_core.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

use my_api::root_function;
use my_api::utils::util;

fn main() {
    let _ = root_function();
    let _ = my_api::root_function();
    let _ = my_api::utils::utils_helper();
    let _ = util::util_mod_helper();
    let _ = my_api::core::core_fn();
}
