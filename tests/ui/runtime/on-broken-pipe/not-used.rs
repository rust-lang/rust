//@ run-pass
//@ aux-build:sigpipe-utils.rs
//@ only-unix because SIGPIPE is a unix thing
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows

fn main() {
    extern crate sigpipe_utils;

    // SIGPIPE shall be ignored since `OnBrokenPipe` is not used
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
