//@ compile-flags: -Copt-level=3
//@ only-64bit (because we're using [ui]size)

#![crate_type = "lib"]
#![feature(slice_from_ptr_range)]

// This is intentionally using a non-power-of-two array length,
// as that's where the optimization differences show up

// CHECK-LABEL: @flatten_via_ptr_range
#[no_mangle]
pub fn flatten_via_ptr_range(slice_of_arrays: &[[i32; 13]]) -> &[i32] {
    // CHECK-NOT: lshr
    // CHECK-NOT: udiv
    // CHECK: mul nuw nsw i64 %{{.+}}, 13
    // CHECK-NOT: lshr
    // CHECK-NOT: udiv
    let r = slice_of_arrays.as_ptr_range();
    let r = r.start.cast()..r.end.cast();
    unsafe { core::slice::from_ptr_range(r) }
}
