//~ ERROR can't find crate for `std`
//~| NOTE can't find crate
//~| NOTE target may not be installed
//~| HELP consider building the standard library from source with `cargo build -Zbuild-std`
//~| HELP consider downloading the target with

//@ compile-flags: --target x86_64-unknown-linux-gnu -Z implicit-sysroot-deps=false
//@ needs-llvm-components: x86
fn main() {}
