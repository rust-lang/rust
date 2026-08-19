//@ add-minicore
//@ revisions: aarch64 android
//@[aarch64] compile-flags: --target aarch64-unknown-none -Tfixed-x18 -Tsanitizer=shadow-call-stack -Zunstable-options
//@[aarch64] needs-llvm-components: aarch64
//@[android] compile-flags: --target aarch64-linux-android -Tsanitizer=shadow-call-stack -Zunstable-options
//@[android] needs-llvm-components: aarch64

#![allow(internal_features)]
#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: ; Function Attrs:{{.*}}shadowcallstack
#[no_mangle]
pub fn foo() {}

// CHECK: attributes #0 = {{.*}}shadowcallstack{{.*}}
