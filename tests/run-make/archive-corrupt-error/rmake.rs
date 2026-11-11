// Regression test for https://github.com/rust-lang/rust/issues/148217
// A corrupt archive with member offset exceeding file boundary should produce
// an error, not an ICE.

//@ ignore-cross-compile

use run_make_support::{path, rfs, rustc, static_lib_name};

fn main() {
    rfs::create_dir("archive");
    rfs::copy("corrupt.a", path("archive").join(static_lib_name("corrupt")));
    rustc()
        .input("lib.rs")
        .crate_type("rlib")
        .library_search_path("archive")
        .arg("-lstatic=corrupt")
        .run_fail()
        .assert_stderr_not_contains("panicked")
        .assert_stderr_not_contains("unexpectedly panicked")
        .assert_stderr_contains("invalid archive member");
}
