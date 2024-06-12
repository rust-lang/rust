//@ ignore-cross-compile

use run_make_support::{cwd, dynamic_lib_name, fs_wrapper, read_dir, run, run_fail, rustc};

fn main() {
    rustc().input("bar.rs").crate_type("dylib").crate_type("rlib").arg("-Cprefer-dynamic").run();
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();

    run("foo");

    fs_wrapper::remove_file(dynamic_lib_name("bar"));
    // This time the command should fail.
    run_fail("foo");
}
