// Passing invalid files to -Z ls (which lists the symbols
// defined by a library crate) used to cause a segmentation fault.
// As this was fixed in #11262, this test checks that no segfault
// occurs when passing the invalid file `bar` to -Z ls.
// See https://github.com/rust-lang/rust/issues/11259

//@ ignore-cross-compile

use run_make_support::fs_wrapper;
use run_make_support::{rmake_out_path, rustc};

fn main() {
    rustc().input("foo.rs");
    rustc().arg("-Zls=root").input(rmake_out_path("foo"));
    fs_wrapper::create_file(rmake_out_path("bar"));
    rustc().arg("-Zls=root").input(rmake_out_path("bar"));
}
