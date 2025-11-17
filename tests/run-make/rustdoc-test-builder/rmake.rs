// This test validates the `--test-builder` rustdoc option.
// It ensures that:
// 1. When the test-builder path points to a non-executable file, rustdoc gracefully fails
// 2. When the test-builder path points to a valid executable, it receives rustc arguments

//@ needs-target-std

use run_make_support::{bare_rustc, path, rfs, rustc_path, rustdoc, target};

fn main() {
    // Test 1: Verify that a non-executable test-builder fails gracefully
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

    // Some targets (for example wasm) cannot execute doctests directly even with a runner,
    // so only exercise the success path when the target can run on the host.
    if target().contains("wasm")
        || target().contains("sgx")
        || std::env::var_os("REMOTE_TEST_CLIENT").is_some()
    {
        return;
    }

    // Test 2: Verify that a valid test-builder is invoked with correct arguments
    // Build a custom test-builder that logs its arguments and forwards to rustc.
    // Use `bare_rustc` so we compile for the host architecture even in cross builds.
    let builder_bin = path("builder-bin");
    bare_rustc().input("builder.rs").output(&builder_bin).run();

    let log_path = path("builder.log");
    let _ = std::fs::remove_file(&log_path);

    // Run rustdoc with our custom test-builder
    rustdoc()
        .input("doctest.rs")
        .arg("--test")
        .arg("-Zunstable-options")
        .arg("--test-builder")
        .arg(&builder_bin)
        .env("REAL_RUSTC", rustc_path())
        .env("BUILDER_LOG", &log_path)
        .run();

    // Verify the custom builder was invoked with rustc-style arguments
    let log_contents = rfs::read_to_string(&log_path);
    assert!(
        log_contents.contains("--crate-type"),
        "expected builder to receive rustc arguments, got:\n{log_contents}"
    );
}
