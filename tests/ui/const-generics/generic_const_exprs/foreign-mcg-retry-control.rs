// Ordinary cross-crate anonymous constants do not require a GCE retry.

//@ edition: 2024
//@ compile-flags: -Znext-solver=globally --crate-type=lib
//@ aux-crate: mcg_dependency=foreign-mcg-retry-control-aux.rs
//@ check-pass

pub fn first(x: mcg_dependency::Arr) -> u8 {
    x[0]
}
