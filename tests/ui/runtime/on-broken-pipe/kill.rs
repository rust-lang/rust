//@ run-pass
//@ aux-build:sigpipe-utils.rs
//@ only-unix because SIGPIPE is a unix thing
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows

// FIXME: Make this work
//@ ignore-backends: gcc

// FIXME: Should not be needed  Create specific issue!!
//@ no-prefer-dynamic

#![feature(on_broken_pipe)]
#![feature(extern_item_impls)]

#[std::io::on_broken_pipe]
fn on_broken_pipe() -> std::io::OnBrokenPipe {
    std::io::OnBrokenPipe::Kill
}

fn main() {
    extern crate sigpipe_utils;

    // `OnBrokenPipe::Kill` is active, so SIGPIPE shall NOT be ignored, instead
    // the default handler shall be installed
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Default);
}
