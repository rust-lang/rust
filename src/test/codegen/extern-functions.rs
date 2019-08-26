// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(unwind_attributes)]

extern {
// CHECK: Function Attrs: nounwind
// CHECK-NEXT: declare void @extern_fn
    fn extern_fn(); // assumed not to unwind
// CHECK-NOT: nounwind
// CHECK: declare void @unwinding_extern_fn
    #[unwind(allowed)]
    fn unwinding_extern_fn();
// CHECK-NOT: nounwind
// CHECK: declare void @aborting_extern_fn
    #[unwind(aborts)]
    fn aborting_extern_fn(); // FIXME: we don't have the attribute here
}

extern "Rust" {
// CHECK-NOT: nounwind
// CHECK: declare void @rust_extern_fn
    fn rust_extern_fn();
// CHECK-NOT: nounwind
// CHECK: declare void @rust_unwinding_extern_fn
    #[unwind(allowed)]
    fn rust_unwinding_extern_fn();
// CHECK-NOT: nounwind
// CHECK: declare void @rust_aborting_extern_fn
    #[unwind(aborts)]
    fn rust_aborting_extern_fn(); // FIXME: we don't have the attribute here
}

pub unsafe fn force_declare() {
    extern_fn();
    unwinding_extern_fn();
    aborting_extern_fn();
    rust_extern_fn();
    rust_unwinding_extern_fn();
    rust_aborting_extern_fn();
}
