// This test ensures that a temporary doctest directory is removed as soon as the doctest
// was run to prevent "No space left on device". To ensure it works, we use a file with two
// doctests, run without the "merged doctests" feature (hence the "2021" edition) with
// one thread to prevent potential concurrent cleanups.
// Regression test for <https://github.com/rust-lang/rust/issues/164145>.

//@ ignore-cross-compile
//@ only-linux

use run_make_support::{is_windows, path, rfs, rustdoc};

fn main() {
    let tmp = path("tmp");
    rfs::create_dir(&tmp);

    let mut cmd = rustdoc()
        .env("TMPDIR", &tmp)
        .input("lib.rs")
        .edition("2021")
        .arg("--test")
        .arg("--test-args=--test-threads=1")
        .run();
}
