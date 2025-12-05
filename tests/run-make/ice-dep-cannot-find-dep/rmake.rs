// This test case creates a situation where the crate loader would run
// into an ICE (internal compiler error) when confronted with an invalid setup where it cannot
// find the dependency of a direct dependency.
//
// The test case makes sure that the compiler produces the expected
// error message but does not ICE immediately after.
//
// See https://github.com/rust-lang/rust/issues/83045

//@ only-x86_64
//@ only-linux
//@ ignore-cross-compile
// Reason: This is a platform-independent issue, no need to waste time testing
// everywhere.

// NOTE: We use `bare_rustc` below so that the compiler can't find liba.rlib
//       If we used `rustc` the additional '-L rmake_out' option would allow rustc to
//       actually find the crate.

use run_make_support::{bare_rustc, rust_lib_name, rustc};

fn main() {
    rustc().crate_name("a").crate_type("rlib").input("a.rs").arg("--verbose").run();
    rustc()
        .crate_name("b")
        .crate_type("rlib")
        .extern_("a", rust_lib_name("a"))
        .input("b.rs")
        .arg("--verbose")
        .run();
    bare_rustc()
        .extern_("b", rust_lib_name("b"))
        .crate_type("rlib")
        .edition("2018")
        .input("c.rs")
        .run_fail()
        .assert_stderr_contains("E0463")
        .assert_stderr_not_contains("internal compiler error");
}
