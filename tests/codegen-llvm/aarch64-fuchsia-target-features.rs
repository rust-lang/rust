//@ add-minicore
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-fuchsia
//@ needs-llvm-components: aarch64

// CHECK: attributes #0 = { {{.*}}"target-features"="{{.*}}+fix-cortex-a53-835769{{.*}}" }

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn test() {}
