//@ run-pass
//@ aux-build:sigpipe-utils.rs
//@ compile-flags: -Zon-broken-pipe=error

fn main() {
    extern crate sigpipe_utils;

    // `-Zon-broken-pipe=error` is active, so we expect SIGPIPE to be ignored.
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
