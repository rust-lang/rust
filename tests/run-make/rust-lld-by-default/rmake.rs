// Ensure that rust-lld is used as the default linker on `x86_64-unknown-linux-gnu`, and that it can
// also be turned off with a CLI flag.

//@ needs-rust-lld
//@ only-x86_64-unknown-linux-gnu

use run_make_support::regex::Regex;
use run_make_support::rustc;
use std::process::Output;

fn main() {
    // A regular compilation should use rust-lld by default. We'll check that by asking the linker
    // to display its version number with a link-arg.
    let output = rustc()
        .env("RUSTC_LOG", "rustc_codegen_ssa::back::link=info")
        .link_arg("-Wl,-v")
        .input("main.rs")
        .run();
    assert!(
        find_lld_version_in_logs(&output),
        "the LLD version string should be present in the output logs:\n{}",
        std::str::from_utf8(&output.stderr).unwrap()
    );

    // But it can still be disabled by turning the linker feature off.
    let output = rustc()
        .env("RUSTC_LOG", "rustc_codegen_ssa::back::link=info")
        .link_arg("-Wl,-v")
        .arg("-Zlinker-features=-lld")
        .input("main.rs")
        .run();
    assert!(
        !find_lld_version_in_logs(&output),
        "the LLD version string should not be present in the output logs:\n{}",
        std::str::from_utf8(&output.stderr).unwrap()
    );
}

fn find_lld_version_in_logs(output: &Output) -> bool {
    let lld_version_re = Regex::new(r"^LLD [0-9]+\.[0-9]+\.[0-9]+").unwrap();
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    stderr.lines().any(|line| lld_version_re.is_match(line.trim()))
}
