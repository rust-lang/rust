// Ensure that -Ctarget-feature takes precedence over -Zllvm-target-feature
//@ add-minicore
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ compile-flags: -Ctarget-feature=+bmi2
//@ compile-flags: -Zllvm-target-feature=-bmi2

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn banana() {
    // CHECK-LABEL: @banana()
    // CHECK-SAME: [[BANANAATTRS:#[0-9]+]]
}
// Test that bmi2 is set both for LLVM and in `cfg(target_feature)`

// CHECK: attributes [[BANANAATTRS]]
// CHECK-SAME: -bmi2,+bmi2

#[cfg(not(target_feature = "bmi2"))]
compile_error!("the bmi2 cfg did not get set");
