// run-pass
// aux-build:sigpipe-utils.rs

#![feature(unix_sigpipe)]
#![feature(rustc_attrs)]

#[unix_sigpipe = "sig_dfl"]
#[rustc_main]
fn rustc_main() {
    extern crate sigpipe_utils;

    // #[unix_sigpipe = "sig_dfl"] is active, so SIGPIPE handler shall be
    // SIG_DFL. Note that we have a #[rustc_main], but it should still work.
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Default);
}
