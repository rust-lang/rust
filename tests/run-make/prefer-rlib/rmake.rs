// Check that `foo.rs` prefers to link to `bar` statically, and can be executed even if the `bar`
// library artifacts are removed.

//@ ignore-cross-compile

use run_make_support::{dynamic_lib_name, path, run, rust_lib_name, rustc};
use std::fs::remove_file;

fn main() {
    rustc().input("bar.rs").crate_type("dylib").crate_type("rlib").run();
    assert!(path(rust_lib_name("bar")).exists());
    rustc().input("foo.rs").run();
    remove_file(rust_lib_name("bar")).unwrap();
    remove_file(dynamic_lib_name("bar")).unwrap();
    run("foo");
}
