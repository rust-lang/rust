//@ only-linux
//@ ignore-wasm32
//@ ignore-wasm64
// ignore-tidy-linelength

// Ensure that on stable we don't suggest restricting with an unsafe trait and we continue
// mentioning the rest of the obligation chain.

use run_make_support::{rust_lib_name, rustc};

fn main() {
    rustc()
        .env("RUSTC_BOOTSTRAP", "-1")
        .input("missing-bound.rs")
        .run_fail()
        .assert_stderr_not_contains("help: consider restricting type parameter `T`")
        .assert_stderr_contains(
            r#"
  = note: required for `std::ops::Range<T>` to implement `Iterator`
  = note: required for `std::ops::Range<T>` to implement `IntoIterator`"#,
        );
}
