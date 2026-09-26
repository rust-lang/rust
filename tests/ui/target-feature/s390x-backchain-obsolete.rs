//@ add-minicore
//@ compile-flags: --crate-type=rlib --target=s390x-unknown-linux-gnu -Ctarget-feature=+backchain
//@ needs-llvm-components: systemz
//@ build-pass

#![feature(s390x_target_feature)]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn test() {}

//~? WARN unstable feature specified for `-Ctarget-feature`: `backchain`
//~? WARN use of `-Ctarget-feature=+backchain` is deprecated
