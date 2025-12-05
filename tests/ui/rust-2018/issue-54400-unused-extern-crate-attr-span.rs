//@ aux-build:edition-lint-paths.rs
//@ run-rustfix
//@ compile-flags:--extern edition_lint_paths
//@ edition:2018

#![deny(rust_2018_idioms)]
#![allow(dead_code, unexpected_cfgs)]

// The suggestion span should include the attribute.

#[cfg(not(FALSE))] //~ HELP remove
extern crate edition_lint_paths;
//~^ ERROR unused extern crate

fn main() {}
