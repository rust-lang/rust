//! Test that static data from external crates can be imported on MSVC targets.
//!
//! On Windows MSVC targets, static data from external rlibs must be imported
//! through `__imp_<symbol>` stubs to ensure proper linking. Without this,
//! the linker would fail with "unresolved external symbol" errors when trying
//! to reference static data from another crate.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/26591>.
//! Fixed in <https://github.com/rust-lang/rust/pull/28646>.

//@ run-pass
//@ aux-build:msvc-static-data-import-lib.rs

extern crate msvc_static_data_import_lib;

fn main() {
    println!("The answer is {}!", msvc_static_data_import_lib::FOO);
}
