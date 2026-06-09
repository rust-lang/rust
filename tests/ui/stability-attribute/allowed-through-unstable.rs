// Test for new `#[rustc_allowed_through_unstable_modules]` attribute
//
//@ aux-build:allowed-through-unstable-core.rs
#![crate_type = "lib"]

extern crate allowed_through_unstable_core;

use allowed_through_unstable_core::unstable_module::OldStableTraitAllowedThoughUnstable; //~WARN use of deprecated module `allowed_through_unstable_core::unstable_module`: use the new path instead
use allowed_through_unstable_core::unstable_module::NewStableTraitNotAllowedThroughUnstable; //~ ERROR use of unstable library feature `unstable_test_feature`
