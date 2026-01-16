//@ ignore-cross-compile because aux-bin does not yet support it
//@ ignore-remote because aux-bin does not yet support it
//@ only-unix because SIGPIPE is a unix thing
//@ aux-bin: assert-inherit-sig_dfl.rs
//@ aux-bin: assert-inherit-sig_ign.rs
//@ run-pass
//@ only-unix because SIGPIPE is a unix thing
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows

// FIXME: Should not be needed  Create specific issue!!
//@ no-prefer-dynamic

#![feature(on_broken_pipe)]
#![feature(rustc_private)]

extern crate libc;

// By default the Rust runtime resets SIGPIPE to SIG_DFL before exec'ing child
// processes so opt-out of that with `-Zon-broken-pipe=kill`. See
// https://github.com/rust-lang/rust/blob/bf4de3a874753bbee3323081c8b0c133444fed2d/library/std/src/sys/pal/unix/process/process_unix.rs#L359-L384
#[std::io::on_broken_pipe]
fn on_broken_pipe() -> std::io::OnBrokenPipe {
    std::io::OnBrokenPipe::Kill
}

fn main() {
    // First expect SIG_DFL in a child process with -`Zon-broken-pipe=inherit`.
    assert_inherit_sigpipe_disposition("auxiliary/bin/assert-inherit-sig_dfl");

    // With SIG_IGN we expect `OnBrokenPipe::Inherit` to also get SIG_IGN.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
    assert_inherit_sigpipe_disposition("auxiliary/bin/assert-inherit-sig_ign");
}

fn assert_inherit_sigpipe_disposition(aux_bin: &str) {
    let mut cmd = std::process::Command::new(aux_bin);
    assert!(cmd.status().unwrap().success());
}

// FIXME: We must use feature flag even if std enables eii
