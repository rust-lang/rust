//@ compile-flags: -C no-prepopulate-passes
//@ needs-unwind

#![crate_type = "lib"]

extern "C" {
    fn extern_fn();
}

extern "C-unwind" {
    fn c_unwind_extern_fn();
}

// The attributes are at the call sites, not the declaration.

// CHECK-LABEL: @force_declare
#[no_mangle]
pub unsafe fn force_declare() {
    // Attributes with `nounwind` here (also see check below).
    // CHECK: call void @extern_fn() #1
    extern_fn();
    // No attributes here.
    // CHECK: call void @c_unwind_extern_fn()
    c_unwind_extern_fn();
}

// CHECK: attributes #1
// CHECK-SAME: nounwind
