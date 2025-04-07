// Test linking using `cc` with `rust-lld`, using the unstable CLI described in MCP 510
// see https://github.com/rust-lang/compiler-team/issues/510 for more info

//@ needs-rust-lld
//@ ignore-s390x lld does not yet support s390x as target

use run_make_support::linker::{assert_rustc_doesnt_use_lld, assert_rustc_uses_lld};
use run_make_support::rustc;

fn main() {
    // Opt-in to lld and the self-contained linker, to link with rust-lld. We'll check that by
    // asking the linker to display its version number with a link-arg.
    assert_rustc_uses_lld(
        rustc()
            .arg("-Zlinker-features=+lld")
            .arg("-Clink-self-contained=+linker")
            .arg("-Zunstable-options")
            .input("main.rs"),
    );

    // It should not be used when we explicitly opt out of lld.
    assert_rustc_doesnt_use_lld(rustc().arg("-Zlinker-features=-lld").input("main.rs"));

    // While we're here, also check that the last linker feature flag "wins" when passed multiple
    // times to rustc.
    assert_rustc_uses_lld(
        rustc()
            .arg("-Clink-self-contained=+linker")
            .arg("-Zunstable-options")
            .arg("-Zlinker-features=-lld")
            .arg("-Zlinker-features=+lld")
            .arg("-Zlinker-features=-lld,+lld")
            .input("main.rs"),
    );
}
