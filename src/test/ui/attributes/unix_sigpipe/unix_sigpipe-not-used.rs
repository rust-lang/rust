// run-pass
// aux-build:sigpipe-utils.rs

fn main() {
    extern crate sigpipe_utils;

    // SIGPIPE shall be ignored since #[unix_sigpipe = "..."] is not used
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
