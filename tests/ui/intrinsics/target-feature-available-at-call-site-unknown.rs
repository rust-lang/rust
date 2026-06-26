//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib --emit=llvm-ir
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(core_intrinsics)]

use std::hint::black_box;
use std::intrinsics::target_feature_available_at_call_site;

#[no_mangle]
pub fn check() {
    let _ = target_feature_available_at_call_site("definitely-not-a-feature");
    //~^ ERROR unknown target feature `definitely-not-a-feature`
}
