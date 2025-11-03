// This test ensures that if the rustdoc test binary is not executable, it will
// gracefully fail and not panic.

//@ needs-target-std

use run_make_support::{path, rfs, rustdoc};

fn main() {
    let absolute_path = path("foo.rs").canonicalize().expect("failed to get absolute path");
    let output = rustdoc()
        .input("foo.rs")
        .arg("--test")
        .arg("-Zunstable-options")
        .arg("--test-builder")
        .arg(&absolute_path)
        .run_fail();

    // We check that rustdoc outputs the error correctly...
    output.assert_stdout_contains("Failed to spawn ");
    // ... and that we didn't panic.
    output.assert_not_ice();
}
