#![feature(unix_sigpipe)]

#[unix_sigpipe = "wrong"] //~ error: valid values for `#[unix_sigpipe = "..."]` are `inherit`, `sig_ign`, or `sig_dfl`
fn main() {}
