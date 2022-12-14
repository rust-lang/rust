#![feature(unix_sigpipe)]

#[unix_sigpipe(inherit)] //~ error: malformed `unix_sigpipe` attribute input
fn main() {}
