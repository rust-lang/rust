// Test that the `reserve-x18` target feature is (not) emitted when
// the `-Zfixed-x18` flag is (not) set.

//@ add-core-stubs
//@ revisions: unset set
//@ needs-llvm-components: aarch64
//@ compile-flags: --target aarch64-unknown-none
//@ [set] compile-flags: -Zfixed-x18

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // unset-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+reserve-x18{{.*}} }
    // set: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+reserve-x18{{.*}} }
}
