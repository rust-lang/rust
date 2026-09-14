//@ compile-flags: -Znext-solver=globally
//@ edition: 2021
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/156352>.
// `async extern "system" fn` was ICE-ing with the new solver because
// `improper_ctypes_definitions` normalized the opaque return type all the way
// to a `Coroutine`, which hit an unexpected `bug!`. After the fix the lint
// fires with "opaque types have no C equivalent" instead of crashing.

#![allow(improper_ctypes_definitions)]

async extern "system" fn check_valid_subset(h: usize) {}

fn main() {}
