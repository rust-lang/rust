//@ aux-crate: sigpipe_utils=sigpipe-utils.rs

#![feature(unix_sigpipe)]

#[unix_sigpipe = "inherit"]
fn main() {
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
