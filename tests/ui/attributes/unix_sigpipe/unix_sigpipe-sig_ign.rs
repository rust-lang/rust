//@ revisions: with_feature without_feature
//@ [with_feature]run-pass
//@ [without_feature]check-fail
//@ aux-build:sigpipe-utils.rs

#![cfg_attr(with_feature, feature(unix_sigpipe))]

#[unix_sigpipe = "sig_ign"] //[without_feature]~ the `#[unix_sigpipe = "sig_ign"]` attribute is an experimental feature
fn main() {
    extern crate sigpipe_utils;

    // #[unix_sigpipe = "sig_ign"] is active, so the legacy behavior of ignoring
    // SIGPIPE shall be in effect
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Ignore);
}
