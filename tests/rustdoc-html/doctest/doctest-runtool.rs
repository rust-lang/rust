// Tests that the --test-runtool argument works.

//@ ignore-cross-compile
//@ aux-bin: doctest-runtool.rs
//@ compile-flags: --test
//@ compile-flags: --test-runtool=auxiliary/bin/doctest-runtool
//@ compile-flags: --test-runtool-arg=arg1 --test-runtool-arg
//@ compile-flags: 'arg2 with space'

/// ```
/// assert_eq!(std::env::var("DOCTEST_RUNTOOL_CHECK"), Ok("xyz".to_string()));
/// ```
pub fn main() {}
