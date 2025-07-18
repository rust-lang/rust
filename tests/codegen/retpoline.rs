// ignore-tidy-linelength
// Test that the
// `retpoline-external-thunk`, `retpoline-indirect-branches`, `retpoline-indirect-calls`
// target features are (not) emitted when the `retpoline/retpoline-external-thunk` flag is (not) set.

//@ add-core-stubs
//@ revisions: disabled enabled_retpoline enabled_retpoline_external_thunk
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [enabled_retpoline] compile-flags: -Zretpoline
//@ [enabled_retpoline_external_thunk] compile-flags: -Zretpoline-external-thunk
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]
extern crate minicore;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // disabled-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-external-thunk{{.*}} }
    // disabled-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-indirect-branches{{.*}} }
    // disabled-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-indirect-calls{{.*}} }

    // enabled_retpoline: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-indirect-branches,+retpoline-indirect-calls{{.*}} }
    // enabled_retpoline_external_thunk: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-external-thunk,+retpoline-indirect-branches,+retpoline-indirect-calls{{.*}} }
}
