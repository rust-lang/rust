//! Ensure that `#[optimize(none)]` functions are never inlined

//@ compile-flags: -Copt-level=3

#![feature(optimize_attribute)]

#[optimize(none)]
pub fn foo() {
    let _x = 123;
}

// CHECK-LABEL: define{{.*}}void @bar
// CHECK: start:
// CHECK: {{.*}}call {{.*}}void
// CHECK: ret void
#[no_mangle]
pub fn bar() {
    foo();
}

fn main() {}
