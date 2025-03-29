// ignore-tidy-linelength
// Test that the
// `retpoline-external-thunk`, `retpoline-indirect-branches`, `retpoline-indirect-calls`
// target features are (not) emitted when the `x86-retpoline` flag is (not) set.

//@ revisions: disabled enabled
//@ needs-llvm-components: x86
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ [enabled] compile-flags: -Zx86-retpoline

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0

    // disabled-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-external-thunk{{.*}} }
    // disabled-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-indirect-branches{{.*}} }
    // disabled-NOT: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-indirect-calls{{.*}} }

    // enabled: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+retpoline-external-thunk,+retpoline-indirect-branches,+retpoline-indirect-calls{{.*}} }
}
