//@ no-prefer-dynamic
//@ compile-flags: --target i686-unknown-linux-gnu -Tregparm=2 -Zunstable-options
//@ needs-llvm-components: x86

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
