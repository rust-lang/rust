// Adapted from; https://github.com/rust-lang/rust/issues/164180#issuecomment-6168887401
//@ check-pass
//@ compile-flags: --crate-type lib
//@ aux-build:non_local.rs
//@ edition: 2024

extern crate non_local;

struct Local;

impl non_local::R for Local {
    async fn read<const L: usize>(&mut self) -> [u8; L] {
        [0; L]
    }
}
