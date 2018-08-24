//
// compile-flags: -C no-prepopulate-passes -C force-frame-pointers=y

#![crate_type="lib"]

// CHECK: attributes #{{.*}} "no-frame-pointer-elim"="true"
pub fn foo() {}
