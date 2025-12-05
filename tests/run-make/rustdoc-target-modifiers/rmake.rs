//! Test that target modifiers are taken into account by `rustdoc`.
//!
//! Otherwise, `rustdoc` errors when trying to generate documentation
//! using dependencies (e.g. `core`) that set a target modifier.
//!
//! Please see https://github.com/rust-lang/rust/issues/144521.

use run_make_support::{rustc, rustdoc};

fn main() {
    rustc()
        .input("d.rs")
        .edition("2024")
        .crate_type("rlib")
        .emit("metadata")
        .sysroot("/dev/null")
        .target("aarch64-unknown-none-softfloat")
        .arg("-Zfixed-x18")
        .run();

    rustdoc()
        .input("c.rs")
        .crate_type("rlib")
        .extern_("d", "libd.rmeta")
        .target("aarch64-unknown-none-softfloat")
        .arg("-Zfixed-x18")
        .run();

    rustdoc()
        .input("c.rs")
        .crate_type("rlib")
        .extern_("d", "libd.rmeta")
        .target("aarch64-unknown-none-softfloat")
        .arg("-Zfixed-x18")
        .arg("--test")
        .run();

    rustdoc()
        .input("c.rs")
        .edition("2024")
        .crate_type("rlib")
        .extern_("d", "libd.rmeta")
        .target("aarch64-unknown-none-softfloat")
        .arg("-Zfixed-x18")
        .arg("--test")
        .run();

    // rustdoc --test detects ABI mismatch
    rustdoc()
        .input("c.rs")
        .crate_type("rlib")
        .extern_("d", "libd.rmeta")
        .target("aarch64-unknown-none-softfloat")
        .arg("--test")
        .run_fail()
        .assert_stderr_contains("mixing `-Zfixed-x18` will cause an ABI mismatch");

    // rustdoc --test -Cunsafe-allow-abi-mismatch=... ignores the mismatch
    rustdoc()
        .input("c.rs")
        .crate_type("rlib")
        .extern_("d", "libd.rmeta")
        .target("aarch64-unknown-none-softfloat")
        .arg("--test")
        .arg("-Cunsafe-allow-abi-mismatch=fixed-x18")
        .run();
}
