//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Regression test for #114532.
// Dead code elimination used to fail when a Drop impl contained a panic
// and a potentially-panicking function was called after the value was created.

struct Foo(bool);

impl Drop for Foo {
    fn drop(&mut self) {
        if self.0 {
            return;
        }
        panic!("dead");
    }
}

// CHECK-LABEL: @foo(
// CHECK-NOT: panic
// CHECK-NOT: call void @{{.*}}panicking
// CHECK: call {{.*}} @unknown(
// CHECK-NEXT: ret void
#[no_mangle]
pub fn foo() {
    let _a = Foo(true);
    unsafe {
        unknown(9);
    }
}

extern "Rust" {
    fn unknown(x: i32) -> bool;
}
