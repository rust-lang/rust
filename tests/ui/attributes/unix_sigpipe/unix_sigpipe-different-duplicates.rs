#![feature(unix_sigpipe)]

#[unix_sigpipe = "sig_ign"]
#[unix_sigpipe = "inherit"] //~ error: multiple `unix_sigpipe` attributes
fn main() {}
