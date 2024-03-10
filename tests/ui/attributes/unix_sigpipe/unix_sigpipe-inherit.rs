//@ revisions: with_feature without_feature
//@ [with_feature]run-pass
//@ [without_feature]check-fail
//@ aux-build:sigpipe-utils.rs

#![cfg_attr(with_feature, feature(unix_sigpipe))]

#[unix_sigpipe = "inherit"] //[without_feature]~ ERROR the `#[unix_sigpipe = "inherit"]` attribute is an experimental feature
fn main() {
    extern crate sigpipe_utils;

    // #[unix_sigpipe = "inherit"] is active, so SIGPIPE shall NOT be ignored,
    // instead the default handler shall be installed. (We assume that the
    // process that runs these tests have the default handler.)
    sigpipe_utils::assert_sigpipe_handler(sigpipe_utils::SignalHandler::Default);
}
