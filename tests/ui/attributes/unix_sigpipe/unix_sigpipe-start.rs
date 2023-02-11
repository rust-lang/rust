#![feature(start)]
#![feature(unix_sigpipe)]

#[start]
#[unix_sigpipe = "inherit"] //~ error: `unix_sigpipe` attribute can only be used on `fn main()`
fn custom_start(argc: isize, argv: *const *const u8) -> isize { 0 }
