// test.rs should produce both an rlib and a dylib
// by default. When the crate_type flag is passed and
// forced to dylib, no rlibs should be produced.
// See https://github.com/rust-lang/rust/issues/11573

//@ ignore-cross-compile

use run_make_support::{count_rlibs, remove_dylibs, remove_rlibs, rustc};

fn main() {
    rustc().input("test.rs").run();
    remove_rlibs("test");
    remove_dylibs("test");
    rustc().crate_type("dylib").input("test.rs").run();
    assert_eq!(count_rlibs("test"), 0);
}
