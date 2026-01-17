// Test async function compilation across revisions with private changes.
// Note: async state machines may legitimately require rebuilds when private
// helpers change, as the state machine type captures implementation details.
//
// - rpass1: Initial compilation
// - rpass2: Private async helper changes body
// - rpass3: Private async helper uses nested async block

//@ revisions: rpass1 rpass2 rpass3
//@ aux-build: async_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

extern crate async_dep;

fn main() {
    let _fut = async_dep::public_async_fn();
    let s = async_dep::AsyncStruct;
    let _fut2 = s.method();
}
