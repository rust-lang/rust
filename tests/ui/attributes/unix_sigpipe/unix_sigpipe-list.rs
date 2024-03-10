//@ revisions: with_feature without_feature

#![cfg_attr(with_feature, feature(unix_sigpipe))]

#[unix_sigpipe(sig_dfl)] //~ error: malformed `unix_sigpipe` attribute input
fn main() {}
