//
// no-system-llvm
// compile-flags: -O

#![crate_type="lib"]

struct A;

impl Drop for A {
    fn drop(&mut self) {
        extern { fn foo(); }
        unsafe { foo(); }
    }
}

#[no_mangle]
pub fn a(a: Box<i32>) {
    // CHECK-LABEL: define void @a
    // CHECK: call void @__rust_dealloc
    // CHECK-NEXT: call void @foo
    let _a = A;
    drop(a);
}
