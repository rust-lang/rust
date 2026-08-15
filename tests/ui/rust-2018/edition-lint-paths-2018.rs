//@ build-pass (FIXME(62277): could be check-pass?)
//@ edition:2018
//@ compile-flags:--extern edition_lint_paths
//@ aux-build:edition-lint-paths.rs

#![deny(absolute_paths_not_starting_with_crate)]

edition_lint_paths::macro_2015!(); // OK

fn main() {}
