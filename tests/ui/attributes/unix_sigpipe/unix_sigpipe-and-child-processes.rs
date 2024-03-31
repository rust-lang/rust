//@ revisions: default sig_dfl sig_ign inherit
//@ ignore-cross-compile because aux-bin does not yet support it
//@ only-unix because SIGPIPE is a unix thing
//@ run-pass
//@ aux-bin:assert-sigpipe-disposition.rs
//@ aux-crate:sigpipe_utils=sigpipe-utils.rs

// Checks the signal disposition of `SIGPIPE` in child processes, and in our own
// process for robustness. Without any `unix_sigpipe` attribute, `SIG_IGN` is
// the default. But there is a difference in how `SIGPIPE` is treated in child
// processes with and without the attribute. Search for
// `unix_sigpipe_attr_specified()` in the code base to learn more.

#![feature(rustc_private)]
#![cfg_attr(any(sig_dfl, sig_ign, inherit), feature(unix_sigpipe))]

extern crate libc;
extern crate sigpipe_utils;

use sigpipe_utils::*;

#[cfg_attr(sig_dfl, unix_sigpipe = "sig_dfl")]
#[cfg_attr(sig_ign, unix_sigpipe = "sig_ign")]
#[cfg_attr(inherit, unix_sigpipe = "inherit")]
fn main() {
    // By default we get SIG_IGN but the child gets SIG_DFL through an explicit
    // reset before exec:
    // https://github.com/rust-lang/rust/blob/bf4de3a874753bbee3323081c8b0c133444fed2d/library/std/src/sys/pal/unix/process/process_unix.rs#L363-L384
    #[cfg(default)]
    let (we_expect, child_expects) = (SignalHandler::Ignore, "SIG_DFL");

    // With #[unix_sigpipe = "sig_dfl"] we get SIG_DFL and the child does too
    // without any special code running before exec.
    #[cfg(sig_dfl)]
    let (we_expect, child_expects) = (SignalHandler::Default, "SIG_DFL");

    // With #[unix_sigpipe = "sig_ign"] we get SIG_IGN and the child does too
    // without any special code running before exec.
    #[cfg(sig_ign)]
    let (we_expect, child_expects) = (SignalHandler::Ignore, "SIG_IGN");

    // With #[unix_sigpipe = "inherit"] we get SIG_DFL and the child does too
    // without any special code running before exec.
    #[cfg(inherit)]
    let (we_expect, child_expects) = (SignalHandler::Default, "SIG_DFL");

    assert_sigpipe_handler(we_expect);

    assert!(
        std::process::Command::new("./auxiliary/bin/assert-sigpipe-disposition")
            .arg(child_expects)
            .status()
            .unwrap()
            .success()
    );
}
