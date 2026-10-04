//@ compile-flags: -Znamespaced-crates
//@ edition: 2024
//@ aux-crate:my_api::utils=trailing_super_crate_import_utils.rs

use my_api::crate as crate_alias;
//~^ ERROR: `crate` in paths can only be used in start position

fn main() {}
