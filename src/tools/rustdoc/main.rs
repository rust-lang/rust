// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]

extern crate rustc_driver;

use std::process::ExitCode;

// Override the C allocator in the same way that the `rustc` binary would do.
rustc_driver::override_c_allocator_in_binary!();

fn main() -> ExitCode {
    rustdoc::main()
}
