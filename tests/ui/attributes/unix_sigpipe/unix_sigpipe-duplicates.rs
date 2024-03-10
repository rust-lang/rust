//@ revisions: with_feature without_feature

#![cfg_attr(with_feature, feature(unix_sigpipe))]

#[unix_sigpipe = "sig_dfl"]
#[unix_sigpipe = "sig_dfl"] //~ error: multiple `unix_sigpipe` attributes
fn main() {}
