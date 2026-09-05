//@ aux-build:nounwind.rs
//@ compile-flags: -C no-prepopulate-passes -C panic=abort -C metadata=a
//@ ignore-android

#![crate_type = "lib"]

extern crate nounwind;

#[no_mangle]
pub fn foo() {
    nounwind::bar();
    // CHECK: @foo() #0
    // CHECK: @bar() #0
    // CHECK: attributes #0 = { {{.*}}nounwind{{.*}} }
}
