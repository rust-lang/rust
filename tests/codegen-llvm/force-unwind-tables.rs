//@ compile-flags: -C no-prepopulate-passes -C force-unwind-tables=y -Copt-level=0

#![crate_type = "lib"]

// CHECK: attributes #{{.*}} uwtable
pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 7, !"uwtable", i32 2}
