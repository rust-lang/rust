// This test should fail with `utils_helper` being unresolvable in `my_api::utils`.
// If a crate contains a module that overlaps with a namespaced crate name, then
// the namespaced crate will not be used in name resolution.

//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ aux-crate: my_api=open-ns-mod-my_api.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

fn main() {
    let _ = my_api::utils::root_helper();
    let _ = my_api::utils::utils_helper();
    //~^ ERROR cannot find function `utils_helper` in module `my_api::utils` [E0425]
}
