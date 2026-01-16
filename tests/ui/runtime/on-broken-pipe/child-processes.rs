//@ revisions: default error kill inherit
//@ ignore-cross-compile because aux-bin does not yet support it
//@ ignore-remote because aux-bin does not yet support it
//@ only-unix because SIGPIPE is a unix thing
//@ ignore-backends: gcc
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
//@ run-pass
//@ aux-bin:assert-sigpipe-disposition.rs
//@ aux-crate:sigpipe_utils=sigpipe-utils.rs

// Checks the signal disposition of `SIGPIPE` in child processes, and in our own
// process for robustness.

// FIXME: Should not be needed  Create specific issue!!
//@ no-prefer-dynamic

#![feature(on_broken_pipe)]
#![feature(rustc_private)]

extern crate libc;

#[cfg_attr(not(default), std::io::on_broken_pipe)]
#[allow(unused)]
fn on_broken_pipe() -> std::io::OnBrokenPipe {
    if cfg!(kill) {
        std::io::OnBrokenPipe::Kill
    } else if cfg!(error) {
        std::io::OnBrokenPipe::Error
    } else if cfg!(inherit) {
        std::io::OnBrokenPipe::Inherit
    } else {
        unreachable!()
    }
}

extern crate sigpipe_utils;

use sigpipe_utils::*;

fn main() {
    // By default we get SIG_IGN but the child gets SIG_DFL through an explicit
    // reset before exec:
    // https://github.com/rust-lang/rust/blob/bf4de3a874753bbee3323081c8b0c133444fed2d/library/std/src/sys/pal/unix/process/process_unix.rs#L363-L384
    #[cfg(default)]
    let (we_expect, child_expects) = (SignalHandler::Ignore, "SIG_DFL");

    // We get SIG_DFL and the child does too without any special code running
    // before exec.
    #[cfg(kill)]
    let (we_expect, child_expects) = (SignalHandler::Default, "SIG_DFL");

    // We get SIG_IGN and the child does too without any special code running
    // before exec.
    #[cfg(error)]
    let (we_expect, child_expects) = (SignalHandler::Ignore, "SIG_IGN");

    // We get SIG_DFL and the child does too without any special code running
    // before exec.
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
