//@ aux-crate: sigpipe_utils=sigpipe-utils.rs

#![feature(on_broken_pipe)]
#![feature(extern_item_impls)]

#[std::io::on_broken_pipe]
fn on_broken_pipe() -> std::io::OnBrokenPipe {
    std::io::OnBrokenPipe::Inherit
}

fn main() {
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
