// gate-test-unix_sigpipe

#![crate_type = "bin"]

#[unix_sigpipe = "sig_ign"] //~ the `#[unix_sigpipe = "sig_ign"]` attribute is an experimental feature
fn main () {}
