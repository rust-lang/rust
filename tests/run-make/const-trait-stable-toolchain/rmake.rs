//@ needs-target-std
//
// Test output of const super trait errors in both stable and nightly.
// We don't want to provide suggestions on stable that only make sense in nightly.

use run_make_support::{diff, rustc};

fn main() {
    let out = rustc()
        .input("const-super-trait.rs")
        .env("RUSTC_BOOTSTRAP", "-1")
        .cfg("feature_enabled")
        .run_fail()
        .assert_stderr_not_contains(
            "as `#[const_trait]` to allow it to have `const` implementations",
        )
        .stderr_utf8();
    diff()
        .expected_file("const-super-trait-stable-enabled.stderr")
        .normalize(
            "may not be used on the .* release channel",
            "may not be used on the NIGHTLY release channel",
        )
        .actual_text("(rustc)", &out)
        .run();
    let out = rustc()
        .input("const-super-trait.rs")
        .cfg("feature_enabled")
        .ui_testing()
        .run_fail()
        .assert_stderr_not_contains("enable `#![feature(const_trait_impl)]` in your crate and mark")
        .assert_stderr_contains("as `#[const_trait]` to allow it to have `const` implementations")
        .stderr_utf8();
    diff()
        .expected_file("const-super-trait-nightly-enabled.stderr")
        .actual_text("(rustc)", &out)
        .run();
    let out = rustc()
        .input("const-super-trait.rs")
        .env("RUSTC_BOOTSTRAP", "-1")
        .run_fail()
        .assert_stderr_not_contains("enable `#![feature(const_trait_impl)]` in your crate and mark")
        .assert_stderr_not_contains(
            "as `#[const_trait]` to allow it to have `const` implementations",
        )
        .stderr_utf8();
    diff()
        .expected_file("const-super-trait-stable-disabled.stderr")
        .actual_text("(rustc)", &out)
        .run();
    let out = rustc()
        .input("const-super-trait.rs")
        .ui_testing()
        .run_fail()
        .assert_stderr_contains("enable `#![feature(const_trait_impl)]` in your crate and mark")
        .stderr_utf8();
    diff()
        .expected_file("const-super-trait-nightly-disabled.stderr")
        .actual_text("(rustc)", &out)
        .run();
}
