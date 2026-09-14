// Tests that namespaced crate names are limited to two segments

//@ aux-crate: nscrate::three::segments=open-ns-my_api_utils.rs
//@ compile-flags: -Z namespaced-crates
//@ edition: 2024
//~? ERROR crate name `nscrate::three::segments` passed to `--extern` can have at most two segments.

fn main() {}
