// It is possible, since #64882, to use the --extern flag without an explicit
// path. In the event of two --extern flags, the explicit one with a path will take
// priority, but otherwise, it is a more concise way of fetching specific libraries.
// This test checks that the default priority of explicit extern flags and rlibs is
// respected.
// See https://github.com/rust-lang/rust/pull/64882

//@ ignore-cross-compile
// Reason: the compiled binary is executed

use run_make_support::{dynamic_lib_name, rfs, run, run_fail, rust_lib_name, rustc};

fn main() {
    rustc().input("bar.rs").crate_type("rlib").crate_type("dylib").arg("-Cprefer-dynamic").run();

    // By default, the rlib has priority over the dylib.
    rustc().input("foo.rs").arg("--extern").arg("bar").run();
    rfs::rename(dynamic_lib_name("bar"), "bar.tmp");
    run("foo");
    rfs::rename("bar.tmp", dynamic_lib_name("bar"));

    rustc().input("foo.rs").extern_("bar", rust_lib_name("bar")).arg("--extern").arg("bar").run();
    rfs::rename(dynamic_lib_name("bar"), "bar.tmp");
    run("foo");
    rfs::rename("bar.tmp", dynamic_lib_name("bar"));

    // The first explicit usage of extern overrides the second pathless --extern bar.
    rustc()
        .input("foo.rs")
        .extern_("bar", dynamic_lib_name("bar"))
        .arg("--extern")
        .arg("bar")
        .run();
    rfs::rename(dynamic_lib_name("bar"), "bar.tmp");
    run_fail("foo");
    rfs::rename("bar.tmp", dynamic_lib_name("bar"));

    // With prefer-dynamic, execution fails as it refuses to use the rlib.
    rustc().input("foo.rs").arg("--extern").arg("bar").arg("-Cprefer-dynamic").run();
    rfs::rename(dynamic_lib_name("bar"), "bar.tmp");
    run_fail("foo");
    rfs::rename("bar.tmp", dynamic_lib_name("bar"));
}
