// Verifies that `-Zsanitizer=kernel-hwaddress` enables lifetime markers.

//@ add-minicore
//@ compile-flags: -Zsanitizer=kernel-hwaddress -Copt-level=0
//@ compile-flags: --target aarch64-unknown-none
//@ needs-llvm-components: aarch64

#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: ; khwasan_lifetime_markers::test
// CHECK:       call void @llvm.lifetime.start
// CHECK:       call void @llvm.lifetime.end
pub fn test() {
    let _x = [0u8; 10];
}
