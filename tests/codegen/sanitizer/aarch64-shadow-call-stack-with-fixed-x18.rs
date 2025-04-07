//@ add-core-stubs
//@ revisions: aarch64 android
//@[aarch64] compile-flags: --target aarch64-unknown-none -Zfixed-x18 -Zsanitizer=shadow-call-stack
//@[aarch64] needs-llvm-components: aarch64
//@[android] compile-flags: --target aarch64-linux-android -Zsanitizer=shadow-call-stack
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
