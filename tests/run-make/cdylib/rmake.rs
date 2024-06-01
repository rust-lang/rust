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

use std::fs::remove_file;

use run_make_support::{cc, dynamic_lib, is_msvc, run, rustc, tmp_dir};

fn main() {
    rustc().input("bar.rs").run();
    rustc().input("foo.rs").run();

    if is_msvc() {
        cc().input("foo.c").arg(tmp_dir().join("foo.dll.lib")).out_exe("foo").run();
    } else {
        cc().input("foo.c")
            .arg("-lfoo")
            .output(tmp_dir().join("foo"))
            .library_search_path(tmp_dir())
            .run();
    }

    run("foo");
    remove_file(dynamic_lib("foo")).unwrap();

    rustc().input("foo.rs").arg("-Clto").run();
    run("foo");
}
