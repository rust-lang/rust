#![feature(unix_sigpipe)]

#[unix_sigpipe = "inherit"]
#[unix_sigpipe = "inherit"] //~ error: multiple `unix_sigpipe` attributes
fn main() {}
