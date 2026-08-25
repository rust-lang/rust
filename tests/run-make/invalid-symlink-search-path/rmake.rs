// In this test, the symlink created is invalid (valid relative to the root, but not relatively to
// where it is located), and used to cause an internal compiler error (ICE) when passed as a library
// search path. This was fixed in #26044, and this test checks that the invalid symlink is instead
// simply ignored.
//
// See https://github.com/rust-lang/rust/issues/26006

//@ needs-target-std
//@ needs-symlink
//Reason: symlink requires elevated permission in Windows

use run_make_support::{path, rfs, rustc};

fn main() {
    // We create two libs: `bar` which depends on `foo`. We need to compile `foo` first.
    rfs::create_dir("out");
    rfs::create_dir("out/foo");
    rustc()
        .input("in/foo/lib.rs")
        .crate_name("foo")
        .crate_type("lib")
        .metadata("foo")
        .output("out/foo/libfoo.rlib")
        .run();
    rfs::create_dir_all("out/bar/deps");
    rfs::symlink_file(path("out/foo/libfoo.rlib"), path("out/bar/deps/libfoo.rlib"));

    // Check that the invalid symlink does not cause an ICE
    rustc()
        .input("in/bar/lib.rs")
        .library_search_path("dependency=out/bar/deps")
        .run_fail()
        .assert_exit_code(1)
        .assert_stderr_not_contains("internal compiler error");
}
