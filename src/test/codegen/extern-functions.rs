// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(unwind_attributes)]

extern {
// CHECK: Function Attrs: nounwind
// CHECK-NEXT: declare void @extern_fn
    fn extern_fn();
// CHECK-NOT: Function Attrs: nounwind
// CHECK: declare void @unwinding_extern_fn
    #[unwind(allowed)]
    fn unwinding_extern_fn();
}

pub unsafe fn force_declare() {
    extern_fn();
    unwinding_extern_fn();
}
