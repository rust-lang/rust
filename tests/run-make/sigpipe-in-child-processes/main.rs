// Checks the signal disposition of `SIGPIPE` in child processes, and in our own
// process for robustness. Without any `unix_sigpipe` attribute, `SIG_IGN` is
// the default. But there is a difference in how `SIGPIPE` is treated in child
// processes with and without the attribute. Search for
// `unix_sigpipe_attr_specified()` in the code base to learn more.
//
// Note that there are many other tests for `unix_sigpipe` in
// tests/ui/attributes/unix_sigpipe.

#![feature(rustc_private)]
#![cfg_attr(any(sig_dfl, sig_ign, inherit), feature(unix_sigpipe))]

extern crate libc;

#[cfg_attr(sig_dfl, unix_sigpipe = "sig_dfl")]
#[cfg_attr(sig_ign, unix_sigpipe = "sig_ign")]
#[cfg_attr(inherit, unix_sigpipe = "inherit")]
fn main() {
    // By default, we get SIG_IGN but the child gets SIG_DFL.
    #[cfg(default)]
    let (we_expect, child_expects) = (libc::SIG_IGN, libc::SIG_DFL);

    // With #[unix_sigpipe = "sig_dfl"] we get SIG_DFL and the child does too.
    #[cfg(sig_dfl)]
    let (we_expect, child_expects) = (libc::SIG_DFL, libc::SIG_DFL);

    // With #[unix_sigpipe = "sig_ign"] we get SIG_IGN and the child does too.
    #[cfg(sig_ign)]
    let (we_expect, child_expects) = (libc::SIG_IGN, libc::SIG_IGN);

    // With #[unix_sigpipe = "inherit"] we get SIG_DFL and the child does too.
    #[cfg(inherit)]
    let (we_expect, child_expects) = (libc::SIG_DFL, libc::SIG_DFL);

    let actual = unsafe {
        let mut actual: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut actual);
        actual.sa_sigaction
    };
    assert_eq!(actual, we_expect, "we did not get the SIGPIPE disposition we expect");

    let child_program = std::env::args().nth(1).unwrap();
    let child_expects = match child_expects {
        libc::SIG_DFL => "SIG_DFL",
        libc::SIG_IGN => "SIG_IGN",
        _ => unreachable!(),
    };
    assert!(
        std::process::Command::new(child_program).arg(child_expects).status().unwrap().success()
    );
}
