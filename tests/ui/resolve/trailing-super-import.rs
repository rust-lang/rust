//@ compile-flags: -Znamespaced-crates
//@ edition: 2024
//@ check-fail
//@ aux-crate:my_api::utils=trailing_super_crate_import_utils.rs

use my_api::super as super_alias;
//~^ ERROR: `super` in paths can only be used in start position, after `self`, or after another `super`

fn main() {}
