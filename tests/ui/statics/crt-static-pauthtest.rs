//@ compile-flags: -C target-feature=+crt-static --target aarch64-unknown-linux-pauthtest
//@ needs-llvm-components: aarch64
//@ only-aarch64-unknown-linux-pauthtest


#![feature(no_core)]
#![no_main]

//~? ERROR aarch64-unknown-linux-pauthtest is incompatible with statically linked libc
