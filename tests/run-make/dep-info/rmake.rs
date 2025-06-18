//@ needs-target-std
//
// This is a simple smoke test for rustc's `--emit dep-info` feature. It prints out
// information about dependencies in a Makefile-compatible format, as a `.d` file.
// Note that this test does not check that the `.d` file is Makefile-compatible.

// This test first checks that emitting dep-info disables static analysis, preventing
// compilation of `erroneous.rs` from causing a compilation failure.
// Then, it checks that compilation using the flag is successful in general, even with
// empty source files or source files that contain a whitespace character.

// Finally, it removes one dependency and checks that compilation is still successful.
// See https://github.com/rust-lang/rust/pull/10698

use run_make_support::{rfs, rustc};

fn main() {
    // We're only emitting dep info, so we shouldn't be running static analysis to
    // figure out that this program is erroneous.
    rustc().input("erroneous.rs").emit("dep-info").run();

    rustc().input("lib.rs").emit("dep-info,link").crate_type("lib").run();
    rfs::remove_file("foo.rs");
    rfs::create_file("foo.rs");
    // Compilation should succeed even if `foo.rs` is empty.
    rustc().input("lib.rs").emit("dep-info,link").crate_type("lib").run();

    // Again, with a space in the filename this time around.
    rustc().input("lib_foofoo.rs").emit("dep-info,link").crate_type("lib").run();
    rfs::remove_file("foo foo.rs");
    rfs::create_file("foo foo.rs");
    // Compilation should succeed even if `foo foo.rs` is empty.
    rustc().input("lib_foofoo.rs").emit("dep-info,link").crate_type("lib").run();

    // When a source file is deleted, compilation should still succeed if the library
    // also loses this source file dependency.
    rfs::remove_file("bar.rs");
    rustc().input("lib2.rs").emit("dep-info,link").crate_type("lib").run();
}
