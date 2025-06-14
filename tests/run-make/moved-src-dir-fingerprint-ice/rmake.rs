// A SourceFile created during compilation may have a relative
// path (e.g. if rustc itself is invoked with a relative path).
// When we write out crate metadata, we convert all relative paths
// to absolute paths using the current working directory.
// However, the working directory was previously not included in the crate hash.
// This meant that the crate metadata could change while the crate
// hash remained the same. Among other problems, this could cause a
// fingerprint mismatch ICE, since incremental compilation uses
// the crate metadata hash to determine if a foreign query is green.
// This test checks that we don't get an ICE when the working directory
// (but not the build directory!) changes between compilation
// sessions.
// See https://github.com/rust-lang/rust/issues/85019

//@ needs-target-std

use run_make_support::{rfs, rust_lib_name, rustc};

fn main() {
    rfs::create_dir("incr");
    rfs::create_dir("first_src");
    rfs::create_dir("output");
    rfs::rename("my_lib.rs", "first_src/my_lib.rs");
    rfs::rename("main.rs", "first_src/main.rs");
    // Build from "first_src"
    std::env::set_current_dir("first_src").unwrap();
    rustc().input("my_lib.rs").incremental("incr").crate_type("lib").run();
    rustc().input("main.rs").incremental("incr").extern_("my_lib", rust_lib_name("my_lib")).run();
    std::env::set_current_dir("..").unwrap();
    rfs::rename("first_src", "second_src");
    std::env::set_current_dir("second_src").unwrap();
    // Build from "second_src" - the output and incremental directory remain identical
    rustc().input("my_lib.rs").incremental("incr").crate_type("lib").run();
    rustc().input("main.rs").incremental("incr").extern_("my_lib", rust_lib_name("my_lib")).run();
}
