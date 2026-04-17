//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu -C target-feature=+avx512f -Zinline-mir=no -C no-prepopulate-passes
//@ needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, lang_items, target_feature_inline_always)]
#![no_core]

extern crate minicore;
use minicore::*;

#[inline(always)]
#[target_feature(enable = "sse")]
#[no_mangle]
pub unsafe fn single_target_feature() -> i32 {
    42
}

// `avx512f` is enough here because it implicitly enables `avx`, which in turn
// implies `sse`. That makes the caller compatible with the callee at this
// callsite, so the `alwaysinline` attribute should be emitted on the call.
#[no_mangle]
// CHECK-LABEL: define{{( noundef)?}} i32 @inherits_from_global() unnamed_addr
pub fn inherits_from_global() -> i32 {
    unsafe {
        // CHECK: %_0 = call{{( noundef)?}} i32 @single_target_feature() [[CALL_ATTRS:#[0-9]+]]
        single_target_feature()
    }
}

// CHECK: attributes [[CALL_ATTRS]] = { alwaysinline nounwind }
