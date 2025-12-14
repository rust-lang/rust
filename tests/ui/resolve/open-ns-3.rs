// This test should fail with `utils` being defined multiple times, since open-ns-mod-my_api.rs
// includes a `mod utils` and we also include open-ns-my_api_utils.rs as a namespaced crate at
// my_api::utils.

//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ build-pass

//@ ignore-test FIXME(packages_as_namespaces): feature not implemented yet

fn main() {
    // FIXME test should fail with conflict here, but unsure how to do error annotation
    // in auxiliary crate.
    // let _ = my_api::root_function();
    // let _ = my_api::utils::root_helper();
    let _ = my_api::utils::utils_helper();
}
