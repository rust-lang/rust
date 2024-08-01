//@ run-pass
//@ aux-build:sigpipe-utils.rs
//@ compile-flags: -Zon-broken-pipe=kill
//@ only-unix because SIGPIPE is a unix thing

#![feature(rustc_attrs)]

#[rustc_main]
fn rustc_main() {
    extern crate sigpipe_utils;

    // `-Zon-broken-pipe=kill` is active, so SIGPIPE handler shall be
    // SIG_DFL. Note that we have a #[rustc_main], but it should still work.
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Default);
}
