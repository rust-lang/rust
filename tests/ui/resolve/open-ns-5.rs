// This test makes sure namespaced crates work if we don't use any use statements but instead fully
// qualify all references and we only have a single namespaced crate with no parent crate.

//@ aux-crate: my_api::utils=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//@ build-pass

//@ ignore-test FIXME(packages_as_namespaces): feature not implemented yet

fn main() {
    let _ = my_api::utils::utils_helper();
}
