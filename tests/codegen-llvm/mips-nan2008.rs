// Test that the `nan2008` target feature is (not) emitted when
// the `-Zmips-nan2008` flag is (not) set.

//@ add-minicore
//@ revisions: unset set
//@ needs-llvm-components: mips
//@ compile-flags: --target mips-unknown-linux-gnu
//@ [set] compile-flags: -Zmips-nan2008

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // unset-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+nan2008{{.*}} }
    // set: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+nan2008{{.*}} }
}
