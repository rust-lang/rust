// Test linking using `cc` with `rust-lld`, using a custom target with features described in MCP 510
// see https://github.com/rust-lang/compiler-team/issues/510 for more info:
//
// Starting from the `x86_64-unknown-linux-gnu` target spec, we add the following options:
// - a linker-flavor using lld via a C compiler
// - the self-contained linker component is enabled

//@ needs-rust-lld
//@ only-x86_64-unknown-linux-gnu

use run_make_support::linker::{assert_rustc_doesnt_use_lld, assert_rustc_uses_lld};
use run_make_support::rustc;

fn main() {
    // Compile to a custom target spec with rust-lld enabled by default. We'll check that by asking
    // the linker to display its version number with a link-arg.
    assert_rustc_uses_lld(
        rustc().crate_type("cdylib").target("custom-target.json").input("lib.rs"),
    );

    // But it can also be disabled via linker features.
    assert_rustc_doesnt_use_lld(
        rustc()
            .crate_type("cdylib")
            .target("custom-target.json")
            .arg("-Zlinker-features=-lld")
            .input("lib.rs"),
    );
}
