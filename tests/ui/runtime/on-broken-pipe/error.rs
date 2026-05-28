//@ run-pass
//@ aux-build:sigpipe-utils.rs
//@ compile-flags: -Zon-broken-pipe=error
//@ only-unix because SIGPIPE is a unix thing

fn main() {
    extern crate sigpipe_utils;

    // `-Zon-broken-pipe=error` is active, so we expect SIGPIPE to be ignored.
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
