// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]

use std::process::ExitCode;

fn main() -> ExitCode {
    rustdoc::main()
}
