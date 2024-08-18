//@ compile-flags: -C no-prepopulate-passes
//@ needs-unwind

#![crate_type = "lib"]

// The flag are only at the call sites, not at the declarations.
// CHECK-LABEL: @force_declare
#[no_mangle]
pub unsafe fn force_declare() {
    // CHECK: call void @extern_fn() [[NOUNWIND_ATTR:#[0-9]+]]
    extern_fn();
    // Call without attributes.
    // CHECK: call void @c_unwind_extern_fn() [[UNWIND_ATTR:#[0-9]+]]
    c_unwind_extern_fn();
}

extern "C" {
    // CHECK-NOT: nounwind
    // CHECK: declare{{.*}}void @extern_fn
    fn extern_fn();
}

extern "C-unwind" {
    // CHECK-NOT: nounwind
    // CHECK: declare{{.*}}void @c_unwind_extern_fn
    fn c_unwind_extern_fn();
}

// CHECK: attributes [[NOUNWIND_ATTR]] = { {{.*}}nounwind{{.*}} }
// CHECK: attributes [[UNWIND_ATTR]]
// CHECK-NOT: nounwind
