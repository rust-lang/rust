//@ compile-flags: -C panic=abort

#![crate_type = "lib"]
#![feature(c_unwind)]

extern "C-unwind" {
    fn bar();
}

// CHECK: Function Attrs:{{.*}}nounwind
// CHECK-NEXT: define{{.*}}void @foo
// Handle both legacy and v0 symbol mangling.
// CHECK: call void @{{.*core9panicking19panic_cannot_unwind}}
#[no_mangle]
pub unsafe extern "C" fn foo() {
    bar();
}
