// gate-test-unix_sigpipe

#![crate_type = "bin"]

#[unix_sigpipe = "inherit"] //~ the `#[unix_sigpipe = "inherit"]` attribute is an experimental feature
fn main () {}
