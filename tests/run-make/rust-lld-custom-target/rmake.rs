// Test linking using `cc` with `rust-lld`, using a custom target with features described in MCP 510
// see https://github.com/rust-lang/compiler-team/issues/510 for more info:
//
// Starting from the `x86_64-unknown-linux-gnu` target spec, we add the following options:
// - a linker-flavor using lld via a C compiler
// - the self-contained linker component is enabled

//@ needs-rust-lld
//@ only-x86_64-unknown-linux-gnu

use run_make_support::regex::Regex;
use run_make_support::rustc;

fn main() {
    // Compile to a custom target spec with rust-lld enabled by default. We'll check that by asking
    // the linker to display its version number with a link-arg.
    let output = rustc()
        .crate_type("cdylib")
        .arg("-Wlinker-messages")
        .target("custom-target.json")
        .link_arg("-Wl,-v")
        .input("lib.rs")
        .run();
    assert!(
        find_lld_version_in_logs(output.stderr_utf8()),
        "the LLD version string should be present in the output logs:\n{}",
        output.stderr_utf8()
    );

    // But it can also be disabled via linker features.
    let output = rustc()
        .crate_type("cdylib")
        .arg("-Wlinker-messages")
        .target("custom-target.json")
        .arg("-Zlinker-features=-lld")
        .link_arg("-Wl,-v")
        .input("lib.rs")
        .run();
    assert!(
        !find_lld_version_in_logs(output.stderr_utf8()),
        "the LLD version string should not be present in the output logs:\n{}",
        output.stderr_utf8()
    );
}

fn find_lld_version_in_logs(stderr: String) -> bool {
    let lld_version_re =
        Regex::new(r"^warning: linker stdout: LLD [0-9]+\.[0-9]+\.[0-9]+").unwrap();
    stderr.lines().any(|line| lld_version_re.is_match(line.trim()))
}
