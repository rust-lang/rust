//@ add-minicore
//@ aux-build: disabled_softfloat.rs
//@ aux-build: enabled_softfloat.rs
//@ revisions: disable-softfloat enable-softfloat
//@ check-fail
//@ [enable-softfloat] compile-flags: --target=s390x-unknown-none-softfloat
//@ [enable-softfloat] needs-llvm-components: systemz
//@ [disable-softfloat] compile-flags: --target=s390x-unknown-linux-gnu
//@ [disable-softfloat] needs-llvm-components: systemz


#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate enabled_softfloat;
//[disable-softfloat]~^ ERROR couldn't find crate `enabled_softfloat` with expected target triple s390x-unknown-linux-gnu
extern crate disabled_softfloat;
//[enable-softfloat]~^ ERROR couldn't find crate `disabled_softfloat` with expected target triple s390x-unknown-none-softfloat
