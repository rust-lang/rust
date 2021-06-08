// compile-flags: -C panic=abort

#![crate_type = "lib"]
#![feature(c_unwind)]

extern "C-unwind" {
    fn bar();
}

// CHECK: Function Attrs:{{.*}}nounwind
// CHECK-NEXT: define{{.*}}void @foo
// CHECK: call void @llvm.trap()
#[no_mangle]
pub unsafe extern "C" fn foo() {
    bar();
}
