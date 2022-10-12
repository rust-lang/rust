#![feature(unix_sigpipe)]
#![unix_sigpipe = "inherit"] //~ error: `unix_sigpipe` attribute cannot be used at crate level

fn main() {}
