//@ add-minicore
//@ compile-flags: -Copt-level=3 --crate-type=lib --target=s390x-unknown-linux-gnu
//@ compile-flags: -Cforce-frame-pointers=yes -Zunstable-options
//@ needs-llvm-components: systemz
#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn test_backchain() {
    // CHECK: @test_backchain() unnamed_addr #0
}
// CHECK: attributes #0 = { {{.*}}"backchain"{{.*}} }
