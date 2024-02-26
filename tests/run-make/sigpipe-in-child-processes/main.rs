// Checks the signal disposition of `SIGPIPE` in child processes. Without any
// `unix_sigpipe` attribute, `SIG_IGN` is the default. But there is a
// difference in how `SIGPIPE` is treated in child processes with and without
// the attribute. Search for `unix_sigpipe_attr_specified()` in the code base to
// learn more.

// Note that tests for the attribute that does not involve child processes are
// in tests/ui/attributes/unix_sigpipe.

#![cfg_attr(any(sig_dfl, sig_ign), feature(unix_sigpipe))]

#[cfg_attr(sig_dfl, unix_sigpipe = "sig_dfl")]
#[cfg_attr(sig_ign, unix_sigpipe = "sig_ign")]
fn main() {
    // By default, we get SIG_IGN but the child gets SIG_DFL.
    #[cfg(default)]
    let expected = "SIG_DFL";

    // With #[unix_sigpipe = "sig_dfl"] we get SIG_DFL and the child does too.
    #[cfg(sig_dfl)]
    let expected = "SIG_DFL";

    // With #[unix_sigpipe = "sig_ign"] we get SIG_IGN and the child does too.
    #[cfg(sig_ign)]
    let expected = "SIG_IGN";

    // With #[unix_sigpipe = "inherit"] we get SIG_DFL and the child does too.
    #[cfg(inherit)]
    let expected = "SIG_DFL";

    let child_program = std::env::args().nth(1).unwrap();
    assert!(std::process::Command::new(child_program).arg(expected).status().unwrap().success());
}
