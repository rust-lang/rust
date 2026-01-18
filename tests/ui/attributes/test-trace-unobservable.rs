//@ edition: 2024
//@ proc-macro: tokenstream_iteration.rs
//@ compile-flags: --test
//@ check-pass

use tokenstream_iteration::inspect_stream;

#[test]
#[inspect_stream]
fn foo() {}
