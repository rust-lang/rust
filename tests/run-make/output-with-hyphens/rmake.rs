// Rust files with hyphens in their filename should
// not result in compiled libraries keeping that hyphen -
// it should become an underscore. Only bin executables
// should keep the hyphen. This test ensures that this rule
// remains enforced.
// See https://github.com/rust-lang/rust/pull/23786

//@ ignore-cross-compile

use run_make_support::{bin_name, path, rust_lib_name, rustc};

fn main() {
    rustc().input("foo-bar.rs").crate_type("bin").run();
    assert!(path(bin_name("foo-bar")).exists());
    rustc().input("foo-bar.rs").crate_type("lib").run();
    assert!(path(rust_lib_name("foo_bar")).exists());
}
