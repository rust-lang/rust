//@ compile-flags: -Znamespaced-crates
//@ edition: 2024
//@ check-pass
//@ aux-crate:my_api::utils=namespaced_crate_utils.rs

use alias::utils;
use my_api as alias;

fn main() {}
