// min-llvm-version: 8.0
// compile-flags: -C no-prepopulate-passes -C force-unwind-tables=y

#![crate_type="lib"]

// CHECK: attributes #{{.*}} uwtable
pub fn foo() {}
