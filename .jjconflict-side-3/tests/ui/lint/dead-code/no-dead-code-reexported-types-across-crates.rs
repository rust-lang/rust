//! Regression test to ensure that `dead_code` warning does not get triggered when using re-exported
//! types that are exposed from a different crate
//!
//! Issue: <https://github.com/rust-lang/rust/issues/14421>

//@ check-pass
//@ aux-build:no-dead-code-reexported-types-across-crates.rs

extern crate no_dead_code_reexported_types_across_crates as bug_lib;

use bug_lib::ExposedType;
use bug_lib::new;

pub fn main() {
    let mut x: ExposedType = new();
    x.foo();
}
