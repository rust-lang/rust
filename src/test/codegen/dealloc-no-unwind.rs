// no-system-llvm
// compile-flags: -O

#![crate_type="lib"]

struct A;

impl Drop for A {
    fn drop(&mut self) {
        extern "C" { fn foo(); }
        unsafe { foo(); }
    }
}

#[no_mangle]
pub fn a(b: Box<i32>) {
    // CHECK-LABEL: define{{.*}}void @a
    // CHECK: call void @__rust_dealloc
    // CHECK-NOT: call void @foo
    let a = A;
    drop(b);
    std::mem::forget(a);
}
