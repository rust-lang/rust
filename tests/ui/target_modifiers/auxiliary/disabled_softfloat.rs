//@ add-minicore
//@ no-prefer-dynamic
//@ compile-flags: --target=s390x-unknown-linux-gnu
//@ needs-llvm-components: systemz

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
