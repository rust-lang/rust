// ignore-tidy-linelength
//@ add-core-stubs
//@ revisions: COMPAT INCOMPAT
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu -Copt-level=3
//@ [COMPAT] compile-flags: -Ctarget-feature=+avx2
//@ [INCOMPAT] compile-flags: -Ctarget-feature=-avx2,-avx

// See also tests/assembly/target-feature-multiple.rs
#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    fn peach() -> u32;
}

#[inline]
#[target_feature(enable = "avx")]
#[no_mangle]
pub unsafe fn apple() -> u32 {
    // CHECK-LABEL: @apple()
    // CHECK-SAME: [[APPLEATTRS:#[0-9]+]] {
    // CHECK: {{.*}}call{{.*}}@peach
    peach()
}

// target features same as global
#[no_mangle]
pub unsafe fn banana() -> u32 {
    // CHECK-LABEL: @banana()
    // CHECK-SAME: [[BANANAATTRS:#[0-9]+]] {
    // COMPAT: {{.*}}call{{.*}}@peach
    // INCOMPAT: {{.*}}call{{.*}}@apple
    apple() // Compatible for inline in COMPAT revision and can't be inlined in INCOMPAT
}

// CHECK: attributes [[APPLEATTRS]]
// COMPAT-SAME: "target-features"="+avx,+avx2,{{.*}}"
// INCOMPAT-SAME: "target-features"="{{(-[^,]+,)*}}-avx2{{(,-[^,]+)*}},-avx{{(,-[^,]+)*}},+avx{{(,\+[^,]+)*}}"
// CHECK: attributes [[BANANAATTRS]]
// COMPAT-SAME: "target-features"="+avx,+avx2,{{.*}}"
// INCOMPAT-SAME: "target-features"="{{(-[^,]+,)*}}-avx2{{(,-[^,]+)*}},-avx{{(,-[^,]+)*}}"
