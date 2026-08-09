// This test ensures we are able to compile -Zbuild-std=std with multiple panic strategies.
//
//@ needs-target-std

use run_make_support::tempfile::TempDir;
use run_make_support::{cargo, rfs, CompletedProcess};

fn main() {
    // This is a regression test to ensure that rustc doesn't load `panic_abort`
    // from the sysroot. See rust-lang/cargo#7359
    test("abort").assert_stderr_contains("duplicate lang item in crate `core`: `sized`");

    let result = test("unwind");
    assert!(result.status().success());

    let result = test("immediate-abort");
    assert!(result.status().success());
}

fn test(panic: &'static str) -> CompletedProcess {
    let dir = TempDir::new().unwrap();

    let manifest = manifest(panic);
    rfs::write(dir.path().join("Cargo.toml"), &manifest);
    rfs::write(dir.path().join("main.rs"), "fn main() {}");

    let mut args = vec!["build", "--release", "-Zbuild-std=std"];
    if panic == "immediate-abort" {
        args.push("-Zpanic-immediate-abort");
    }
    cargo()
        .current_dir(dir.path())
        .args(&args)
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run_unchecked()
}

fn manifest(panic: &'static str) -> String {
    format!(
        r#"[package]
name = "foo"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "foo"
path = "main.rs"

[profile.release]
panic = "{panic}"
"#
    )
}
