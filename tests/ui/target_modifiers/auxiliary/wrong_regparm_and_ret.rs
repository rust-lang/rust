//@ no-prefer-dynamic
//@ compile-flags: --target i686-unknown-linux-gnu -Tregparm=2 -Treg-struct-return=true
//@ compile-flags: -Zunstable-options
//@ needs-llvm-components: x86

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
