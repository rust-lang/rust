//@ ignore-auxiliary (used by `./allow-in-other-module.rs`)

// This should not warn.
#![allow(not_a_real_lint)]

// This should not warn, either.
#[allow(not_a_real_lint)]
fn m() {}
