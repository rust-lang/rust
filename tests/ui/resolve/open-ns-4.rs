// This tests that namespaced crates are shadowed.

//@ aux-crate: my_api=open-ns-my_api.rs
//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024

fn main() {
    let _ = my_api::root_function();
    let _ = my_api::utils::utils_helper();
    //~^ ERROR cannot find `utils` in `my_api` [E0433]
}
