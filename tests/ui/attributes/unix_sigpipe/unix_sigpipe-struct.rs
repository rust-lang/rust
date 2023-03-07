#![feature(unix_sigpipe)]

#[unix_sigpipe = "inherit"] //~ error: `unix_sigpipe` attribute can only be used on `fn main()`
struct S;

fn main() {}
