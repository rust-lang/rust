// compile-flags: -C no-prepopulate-passes -C panic=abort -C force-unwind-tables=n
// ignore-windows

#![crate_type="lib"]

// CHECK-NOT: attributes #{{.*}} uwtable
pub fn foo() {}
