#![feature(unix_sigpipe)]
#![unix_sigpipe = "sig_dfl"] //~ error: `unix_sigpipe` attribute cannot be used at crate level

fn main() {}
