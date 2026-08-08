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
//
//@ needs-target-std

use run_make_support::{cargo, path};

fn main() {
    let target_dir = path("target");

    cargo()
        .args(["build", "--manifest-path", "Cargo.toml", "-Zbuild-std=std,core,panic_abort"])
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run();
}
