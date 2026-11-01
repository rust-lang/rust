//@ ignore-cross-compile

use run_make_support::{dynamic_lib_name, rfs, run, run_fail, rustc};

fn main() {
    rustc().input("bar.rs").crate_type("dylib").crate_type("rlib").arg("-Cprefer-dynamic").run();
    rustc().input("foo.rs").arg("-Cprefer-dynamic").run();

    run("foo");

    rfs::remove_file(dynamic_lib_name("bar"));
    // This time the command should fail.
    run_fail("foo");
}
