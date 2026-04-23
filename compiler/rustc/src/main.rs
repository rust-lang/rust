// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]
// Several crates are depended upon but unused so that they are present in the sysroot
#![expect(unused_crate_dependencies)]

use std::process::ExitCode;

fn main() -> ExitCode {
    rustc_driver::main()
}
