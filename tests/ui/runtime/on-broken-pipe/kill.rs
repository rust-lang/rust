//@ run-pass
//@ aux-build:sigpipe-utils.rs
//@ compile-flags: -Zon-broken-pipe=kill
//@ only-unix because SIGPIPE is a unix thing

fn main() {
    extern crate sigpipe_utils;

    // `-Zon-broken-pipe=kill` is active, so SIGPIPE shall NOT be ignored, instead
    // the default handler shall be installed
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Default);
}
