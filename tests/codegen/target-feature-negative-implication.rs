//@ add-core-stubs
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ compile-flags: -Ctarget-feature=-avx2

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn banana() {
    // CHECK-LABEL: @banana()
    // CHECK-SAME: [[BANANAATTRS:#[0-9]+]] {
}

// CHECK: attributes [[BANANAATTRS]]
// CHECK-SAME: -avx512
