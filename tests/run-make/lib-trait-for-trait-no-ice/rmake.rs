// Inside a library, implementing a trait for another trait
// with a lifetime used to cause an internal compiler error (ICE).
// This test checks that this bug does not make a resurgence -
// first by ensuring successful compilation, then verifying that
// the lib crate-type flag was actually followed.
// See https://github.com/rust-lang/rust/issues/18943

use run_make_support::{fs_wrapper, rust_lib_name, rustc};

fn main() {
    rustc().input("foo.rs").crate_type("lib").run();
    fs_wrapper::remove_file(rust_lib_name("foo"));
}
