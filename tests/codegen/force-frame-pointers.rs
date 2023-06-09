// compile-flags: -C no-prepopulate-passes -C force-frame-pointers=y

#![crate_type="lib"]

// CHECK: attributes #{{.*}} "frame-pointer"="all"
pub fn foo() {}
