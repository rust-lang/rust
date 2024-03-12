#![feature(unix_sigpipe)]

#[unix_sigpipe("sig_dfl")] //~ error: malformed `unix_sigpipe` attribute input
fn main() {}
