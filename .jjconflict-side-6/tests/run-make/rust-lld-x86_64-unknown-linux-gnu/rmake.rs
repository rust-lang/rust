// Ensure that rust-lld is used as the default linker on `x86_64-unknown-linux-gnu`
// and that it can also be turned off with a CLI flag.
//
// This version of the test checks that LLD is used by default when LLD is enabled in the
// toolchain. There is a separate test that checks that LLD is used for dist artifacts
// unconditionally.

//@ needs-rust-lld
//@ only-x86_64-unknown-linux-gnu

use run_make_support::linker::{assert_rustc_doesnt_use_lld, assert_rustc_uses_lld};
use run_make_support::rustc;

fn main() {
    // A regular compilation should use rust-lld by default.
    assert_rustc_uses_lld(rustc().input("main.rs"));

    // But it can still be disabled by turning the linker feature off.
    assert_rustc_doesnt_use_lld(rustc().arg("-Clinker-features=-lld").input("main.rs"));
}
