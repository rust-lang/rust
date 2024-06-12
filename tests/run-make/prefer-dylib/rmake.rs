//@ ignore-cross-compile

use run_make_support::{cwd, dynamic_lib_name, read_dir, run, run_fail, rustc};
use std::fs::remove_file;

fn main() {
    rustc().input("bar.rs").crate_type("dylib").crate_type("rlib").arg("-Cprefer-dynamic").run();
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();

    run("foo");

    remove_file(dynamic_lib_name("bar")).unwrap();
    // This time the command should fail.
    run_fail("foo");
}
