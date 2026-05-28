//@ no-prefer-dynamic
//@ compile-flags: --target i686-unknown-linux-gnu -Zreg-struct-return=true
//@ needs-llvm-components: x86

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
