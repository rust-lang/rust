//@ only-x86_64
// FIXME: Once GCC backend is fixed, remove this `ignore-backends`.
//@ ignore-backends: gcc

// Ensure that on stable we don't suggest restricting with an unsafe trait and we continue
// mentioning the rest of the obligation chain.

use run_make_support::{diff, rustc};

fn main() {
    let out = rustc()
        .env("RUSTC_BOOTSTRAP", "-1")
        .input("unstable-target-feature.rs")
        .args(&["-Ctarget-feature=+x87", "--crate-type=rlib"])
        .run()
        .assert_stderr_not_contains("help: consider restricting type parameter `T`")
        .assert_stderr_contains("unstable feature")
        .stderr_utf8();
    diff().expected_file("unstable-target-feature.stderr").actual_text("(stable rustc)", &out).run()
}
