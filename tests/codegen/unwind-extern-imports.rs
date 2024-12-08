//@ compile-flags: -C no-prepopulate-passes
//@ needs-unwind

#![crate_type = "lib"]

extern "C" {
    // CHECK: Function Attrs:{{.*}}nounwind
    // CHECK-NEXT: declare{{.*}}void @extern_fn
    fn extern_fn();
}

extern "C-unwind" {
    // CHECK-NOT: nounwind
    // CHECK: declare{{.*}}void @c_unwind_extern_fn
    fn c_unwind_extern_fn();
}

pub unsafe fn force_declare() {
    extern_fn();
    c_unwind_extern_fn();
}
