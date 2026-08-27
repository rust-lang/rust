// This test ensures we are able to compile -Zbuild-std=std with multiple panic strategies.
//
//@ needs-target-std

use run_make_support::tempfile::TempDir;
use run_make_support::{cargo, rfs};

fn main() {
    // This is a regression test to ensure that rustc doesn't load `panic_abort`
    // from the sysroot. See rust-lang/cargo#7359
    test("abort");

    // The `panic_abort` crate must be compiled with the `-Cpanic=abort` option
    // and the compiler has a check to enforce this. However `build-std`
    // does not yet respect the `std` profile, it may lead to a mismatch:
    // - Cargo profile sets `panic = "unwind"` -> `panic_abort` is not activated (linked).
    // - But `panic_abort` is still compiled with `-Cpanic=unwind`.
    //
    // This test ensures that the check is not triggered in such a situation.
    //
    // FIXME(build-std): ideally, `panic_abort` should always be compiled with
    // `-Cpanic=abort`, even when unused.
    test("unwind");

    test("immediate-abort");
}

fn test(panic: &'static str) {
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
        .run();
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
