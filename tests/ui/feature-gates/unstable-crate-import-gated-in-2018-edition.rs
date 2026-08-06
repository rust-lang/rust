//! Regression test for <https://github.com/rust-lang/rust/issues/52489>
//@ edition:2018
//@ aux-build:unstable-crate-import-gated-in-2018-edition.rs
//@ compile-flags: --extern unstable_crate_import_gated_in_2018_edition

use unstable_crate_import_gated_in_2018_edition;
//~^ ERROR use of unstable library feature `unstable_crate_import_gated_in_2018_edition`

fn main() {}
