//@ revisions: locally_unchecked locally_checked
//@ aux-crate:checked=checked.rs
//@ edition: 2021

// Test that we treat checked type aliases from external crates as checked independently of whether
// the local crate enables feature `checked_type_aliases` or not.

#![cfg_attr(locally_checked, feature(checked_type_aliases))]

fn main() {
    let _: checked::Alias<String>; //~ ERROR the trait bound `String: Copy` is not satisfied
}
