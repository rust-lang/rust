#![feature(unix_sigpipe)]
// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]

#[unix_sigpipe = "sig_dfl"]
fn main() {
    rustdoc::main()
}
