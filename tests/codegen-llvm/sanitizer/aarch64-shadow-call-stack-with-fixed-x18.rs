//@ add-minicore
//@ revisions: aarch64 android
//@[aarch64] compile-flags: --target aarch64-unknown-none -Zfixed-x18
//@[aarch64] needs-llvm-components: aarch64
//@[android] compile-flags: --target aarch64-linux-android
//@[android] needs-llvm-components: aarch64
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=shadow-call-stack

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
