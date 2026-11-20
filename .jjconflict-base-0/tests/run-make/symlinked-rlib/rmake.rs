// Rustc did not recognize libraries which were symlinked
// to files having extension other than .rlib. This was fixed
// in #32828. This test creates a symlink to "foo.xxx", which has
// an unusual file extension, and checks that rustc can successfully
// use it as an rlib library.
// See https://github.com/rust-lang/rust/pull/32828

//@ ignore-cross-compile
//@ needs-symlink

use run_make_support::{cwd, rfs, rustc};

fn main() {
    rustc().input("foo.rs").crate_type("rlib").output("foo.xxx").run();
    rfs::symlink_file("foo.xxx", "libfoo.rlib");
    rustc().input("bar.rs").library_search_path(cwd()).run();
}
