// Regression test for #163791.

//@ edition: 2024
//@ compile-flags: -Znext-solver=globally --crate-type=lib
//@ aux-crate: gce_dependency=foreign-gce-retry-163791.rs
//@ check-pass

pub fn first(x: gce_dependency::Arr) -> u8 {
    x[0]
}
