// Ensure that rust-lld is used as the default linker on `x86_64-unknown-linux-gnu` on the nightly
// channel, and that it can also be turned off with a CLI flag.

//@ needs-rust-lld
//@ ignore-beta
//@ ignore-stable
//@ only-x86_64-unknown-linux-gnu

use run_make_support::linker::{assert_rustc_doesnt_use_lld, assert_rustc_uses_lld};
use run_make_support::rustc;

fn main() {
    // A regular compilation should use rust-lld by default. We'll check that by asking the linker
    // to display its version number with a link-arg.
    assert_rustc_uses_lld(rustc().input("main.rs"));

    // But it can still be disabled by turning the linker feature off.
    assert_rustc_doesnt_use_lld(rustc().arg("-Zlinker-features=-lld").input("main.rs"));
}
