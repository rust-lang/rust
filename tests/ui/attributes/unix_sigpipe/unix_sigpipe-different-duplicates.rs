//@ revisions: with_feature without_feature

#![cfg_attr(with_feature, feature(unix_sigpipe))]

#[unix_sigpipe = "sig_dfl"]
#[unix_sigpipe = "sig_ign"] //[without_feature]~ the `#[unix_sigpipe = "sig_ign"]` attribute is an experimental feature
//~^ error: multiple `unix_sigpipe` attributes
fn main() {}
