// Ensure that rust-lld is used as the default linker on `x86_64-unknown-linux-gnu` on the nightly
// channel, and that it can also be turned off with a CLI flag.

//@ needs-rust-lld
//@ ignore-beta
//@ ignore-stable
//@ only-x86_64-unknown-linux-gnu

use run_make_support::regex::Regex;
use run_make_support::rustc;

fn main() {
    // A regular compilation should use rust-lld by default. We'll check that by asking the linker
    // to display its version number with a link-arg.
    let output = rustc().verbose().link_arg("-Wl,-v").input("main.rs").run();
    assert!(
        find_lld_version_in_logs(output.stderr_utf8()),
        "the LLD version string should be present in the output logs:\n{}",
        output.stderr_utf8()
    );

    // But it can still be disabled by turning the linker feature off.
    let output =
        rustc().verbose().link_arg("-Wl,-v").arg("-Zlinker-features=-lld").input("main.rs").run();
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
