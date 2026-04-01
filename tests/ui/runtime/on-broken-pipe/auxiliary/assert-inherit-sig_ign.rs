//@ aux-crate: sigpipe_utils=sigpipe-utils.rs
//@ compile-flags: -Zon-broken-pipe=inherit

fn main() {
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
