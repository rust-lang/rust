// Check that `foo.rs` prefers to link to `bar` statically, and can be executed even if the `bar`
// library artifacts are removed.

//@ ignore-cross-compile

use run_make_support::{dynamic_lib_name, path, rfs, run, rust_lib_name, rustc};

fn main() {
    rustc().input("bar.rs").crate_type("dylib").crate_type("rlib").run();
    assert!(path(rust_lib_name("bar")).exists());
    rustc().input("foo.rs").run();
    rfs::remove_file(rust_lib_name("bar"));
    rfs::remove_file(dynamic_lib_name("bar"));
    run("foo");
}
