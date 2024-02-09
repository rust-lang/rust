//@ revisions: with_feature without_feature

#![cfg_attr(with_feature, feature(unix_sigpipe))]

#[unix_sigpipe = "wrong"] //~ error: the only valid variant of the `unix_sigpipe` attribute is `#[unix_sigpipe = "sig_dfl"]
fn main() {}
