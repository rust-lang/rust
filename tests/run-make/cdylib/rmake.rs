// This test tries to check that basic cdylib libraries can be compiled and linked successfully
// with C code, that the cdylib itself can depend on another rlib, and that the library can be built
// with LTO.
//
// - `bar.rs` is a rlib
// - `foo.rs` is a cdylib that relies on an extern crate `bar` and defines two `extern "C"`
//   functions:
//     - `foo()` which calls `bar::bar()`.
//     - `bar()` which implements basic addition.

//@ ignore-cross-compile

use run_make_support::{cc, cwd, dynamic_lib_name, is_windows_msvc, rfs, run, rustc};

fn main() {
    rustc().input("bar.rs").run();
    rustc().input("foo.rs").run();

    if is_windows_msvc() {
        cc().input("foo.c").arg("foo.dll.lib").out_exe("foo").run();
    } else {
        cc().input("foo.c").arg("-lfoo").library_search_path(cwd()).output("foo").run();
    }

    run("foo");
    rfs::remove_file(dynamic_lib_name("foo"));

    rustc().input("foo.rs").arg("-Clto").run();
    run("foo");
}
