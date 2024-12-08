//@ compile-flags:--test
//@ check-pass
#![allow(rustdoc::invalid_codeblock_attributes)]

// https://github.com/rust-lang/rust/pull/124577#issuecomment-2276034737

// Test that invalid langstrings don't get run.

/// ```{rust,ignore}
/// panic!();
/// ```
pub struct Foo;
