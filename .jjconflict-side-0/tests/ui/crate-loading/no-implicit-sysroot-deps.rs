//~ ERROR can't find crate for `std`
//~| NOTE can't find crate
//~| NOTE target may not be installed
//~| HELP consider building the standard library from source with `cargo build -Zbuild-std`
//~| HELP consider downloading the target with

//@ compile-flags: --target x86_64-unknown-linux-gnu -Z implicit-sysroot-deps=false
//@ needs-llvm-components: x86

// This program has an implicit dependency on std, injected by rust. This test ensures that rustc
// does not search in the sysroot for it when `-Zimplicit-sysroot-deps` is false, and that an
// error is thrown when std is not available on any other search paths.

fn main() {}
