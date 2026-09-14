//@ revisions: locally_eager locally_lazy
//@ aux-crate:lazy=lazy.rs
//@ edition: 2021

// Test that we treat lazy type aliases from external crates as lazy independently of whether the
// local crate enables `checked_type_aliases` or not.

#![cfg_attr(
    locally_lazy,
    feature(checked_type_aliases),
    allow(incomplete_features)
)]

fn main() {
    let _: lazy::Alias<String>; //~ ERROR the trait bound `String: Copy` is not satisfied
}
