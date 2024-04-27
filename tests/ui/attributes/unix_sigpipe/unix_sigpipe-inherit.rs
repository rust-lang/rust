//@ ignore-cross-compile because aux-bin does not yet support it
//@ only-unix because SIGPIPE is a unix thing
//@ aux-bin: assert-inherit-sig_dfl.rs
//@ aux-bin: assert-inherit-sig_ign.rs
//@ run-pass

#![feature(rustc_private, unix_sigpipe)]

extern crate libc;

// By default the Rust runtime resets SIGPIPE to SIG_DFL before exec:ing child
// processes so opt-out of that with `#[unix_sigpipe = "sig_dfl"]`. See
// https://github.com/rust-lang/rust/blob/bf4de3a874753bbee3323081c8b0c133444fed2d/library/std/src/sys/pal/unix/process/process_unix.rs#L359-L384
#[unix_sigpipe = "sig_dfl"]
fn main() {
    // First expect SIG_DFL in a child process with #[unix_sigpipe = "inherit"].
    assert_inherit_sigpipe_disposition("auxiliary/bin/assert-inherit-sig_dfl");

    // With SIG_IGN we expect #[unix_sigpipe = "inherit"] to also get SIG_IGN.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
    assert_inherit_sigpipe_disposition("auxiliary/bin/assert-inherit-sig_ign");
}

fn assert_inherit_sigpipe_disposition(aux_bin: &str) {
    let mut cmd = std::process::Command::new(aux_bin);
    assert!(cmd.status().unwrap().success());
}
