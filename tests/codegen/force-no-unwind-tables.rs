//@ compile-flags: -C no-prepopulate-passes -C panic=abort -C force-unwind-tables=n
//@ ignore-windows: unwind tables are required for panics on Windows

#![crate_type = "lib"]

// CHECK-LABEL: define{{.*}}void @foo
// CHECK-NOT: attributes #{{.*}} uwtable
#[no_mangle]
fn foo() {
    panic!();
}
