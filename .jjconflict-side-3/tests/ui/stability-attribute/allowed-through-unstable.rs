// Test for `#[rustc_allowed_through_unstable_modules]` attribute
//
//@ aux-build:allowed-through-unstable-core.rs
#![crate_type = "lib"]

extern crate allowed_through_unstable_core;

use allowed_through_unstable_core::unstable_module::OldStableTraitAllowedThoughUnstable; //~WARN use of deprecated import through accidentally stabilized module `unstable_module`
use allowed_through_unstable_core::unstable_module::NewStableTraitNotAllowedThroughUnstable; //~ ERROR use of unstable library feature `unstable_test_feature`
