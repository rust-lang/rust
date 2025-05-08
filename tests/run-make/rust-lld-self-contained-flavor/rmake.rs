// When using an `ld.lld` linker flavor, rustc will invoke lld directly. This test ensures that
// turning on the self-contained linker will result in rustc choosing `rust-lld` instead of the
// system lld.
//
// This is not straigthforward to test, so we make linking fail and look for the linker name
// appearing in the failure.

//@ needs-rust-lld
//@ only-x86_64-unknown-linux-gnu

use run_make_support::{Rustc, rustc};

// Make linking fail because of an incorrect flag. In case there's an issue on CI or locally, we
// also ask for rustc's linking debug logging, in order to have more information in the test logs.
fn make_linking_fail(linker_flavor: &str) -> Rustc {
    let mut rustc = rustc();
    rustc
        .env("RUSTC_LOG", "rustc_codegen_ssa::back::link") // ask for linking debug logs
        .linker_flavor(linker_flavor)
        .link_arg("--baguette") // ensures linking failure
        .input("main.rs");
    rustc
}

fn main() {
    // 1. Using `ld` directly via the linker flavor.
    make_linking_fail("ld").run_fail().assert_stderr_contains("error: linking with `ld` failed");

    // 2. Using `lld` via the linker flavor. We ensure the self-contained linker is disabled to use
    //    the system lld.
    //
    // This could fail in two ways:
    // - the most likely case: `lld` rejects our incorrect link arg
    // - or there may not be an `lld` on the $PATH. The testing/run-make infrastructure runs tests
    //   with llvm tools in the path and there is an `lld` executable there most of the time (via
    //   `ci-llvm`). But since one can turn that off in the config, we also look for the usual
    //   "-fuse-ld=lld" failure.
    let system_lld_failure = make_linking_fail("ld.lld")
        .arg("-Clink-self-contained=-linker")
        .arg("-Zunstable-options")
        .run_fail();
    let lld_stderr = system_lld_failure.stderr_utf8();
    assert!(
        lld_stderr.contains("error: linking with `lld` failed")
            || lld_stderr.contains("error: linker `lld` not found"),
        "couldn't find `lld` failure in stderr: {}",
        lld_stderr,
    );

    // 3. Using the same lld linker flavor and enabling the self-contained linker should use
    //    `rust-lld`.
    make_linking_fail("ld.lld")
        .arg("-Clink-self-contained=+linker")
        .arg("-Zunstable-options")
        .run_fail()
        .assert_stderr_contains("error: linking with `rust-lld` failed");
}
