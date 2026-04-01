//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @loop_skips_vtable_load
#[no_mangle]
pub fn loop_skips_vtable_load(x: &dyn Fn()) {
    // CHECK: load ptr, ptr %0{{.*}}, !invariant.load
    // CHECK-NEXT: tail call void %1
    // CHECK-NOT: load ptr
    x();
    for _ in 0..100 {
        // CHECK: tail call void %1
        x();
    }
}
