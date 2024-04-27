#![feature(unix_sigpipe)]

#[unix_sigpipe] //~ error: malformed `unix_sigpipe` attribute input
fn main() {}
