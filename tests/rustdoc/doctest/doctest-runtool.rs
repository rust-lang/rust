// Tests that the --runtool argument works.

//@ ignore-cross-compile
//@ aux-bin: doctest-runtool.rs
//@ compile-flags: --test
//@ compile-flags: --runtool=auxiliary/bin/doctest-runtool
//@ compile-flags: --runtool-arg=arg1 --runtool-arg
//@ compile-flags: 'arg2 with space'
//@ compile-flags: -Zunstable-options

/// ```
/// assert_eq!(std::env::var("DOCTEST_RUNTOOL_CHECK"), Ok("xyz".to_string()));
/// ```
pub fn main() {}
