// test.rs should produce both an rlib and a dylib
// by default. When the crate_type flag is passed and
// forced to dylib, no rlibs should be produced.
// See https://github.com/rust-lang/rust/issues/11573

//@ ignore-cross-compile

use std::path::Path;

use run_make_support::{
    cwd, dynamic_lib_name, has_extension, rfs, rust_lib_name, rustc, shallow_find_files,
};

fn main() {
    rustc().input("test.rs").run();
    assert!(Path::new(&dynamic_lib_name("test")).exists());
    assert!(Path::new(&rust_lib_name("test")).exists());

    rfs::remove_file(rust_lib_name("test"));
    rustc().crate_type("dylib").input("test.rs").run();
    assert!(shallow_find_files(cwd(), |path| { has_extension(path, "rlib") }).is_empty());
}
