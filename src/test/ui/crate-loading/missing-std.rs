// compile-flags: --target x86_64-unknown-uefi
// needs-llvm-components: x86
// rustc-env:CARGO=/usr/bin/cargo
#![feature(no_core)]
#![no_core]
extern crate core;
//~^ ERROR can't find crate for `core`
//~| NOTE can't find crate
//~| NOTE target may not be installed
//~| HELP consider building the standard library from source with `cargo build -Zbuild-std`
//~| HELP consider downloading the target with `rustup target add x86_64-unknown-uefi`
fn main() {}
