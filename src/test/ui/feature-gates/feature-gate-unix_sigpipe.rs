#![crate_type = "bin"]

#[unix_sigpipe = "inherit"] //~ the `#[unix_sigpipe]` attribute is an experimental feature
fn main () {}
