//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]
#![feature(get_mut_unchecked, new_uninit)]

use std::sync::Arc;

// CHECK-LABEL: @new_from_array
#[no_mangle]
pub fn new_from_array(x: u64) -> Arc<[u64]> {
    // Ensure that we only generate one alloca for the array.

    // CHECK: alloca
    // CHECK-SAME: [8000 x i8]
    // CHECK-NOT: alloca
    let array = [x; 1000];
    Arc::new(array)
}

// CHECK-LABEL: @new_uninit
#[no_mangle]
pub fn new_uninit(x: u64) -> Arc<[u64; 1000]> {
    // CHECK: call alloc::sync::arcinner_layout_for_value_layout
    // CHECK-NOT: call alloc::sync::arcinner_layout_for_value_layout
    let mut arc = Arc::new_uninit();
    unsafe { Arc::get_mut_unchecked(&mut arc) }.write([x; 1000]);
    unsafe { arc.assume_init() }
}

// CHECK-LABEL: @new_uninit_slice
#[no_mangle]
pub fn new_uninit_slice(x: u64) -> Arc<[u64]> {
    // CHECK: call alloc::sync::arcinner_layout_for_value_layout
    // CHECK-NOT: call alloc::sync::arcinner_layout_for_value_layout
    let mut arc = Arc::new_uninit_slice(1000);
    for elem in unsafe { Arc::get_mut_unchecked(&mut arc) } {
        elem.write(x);
    }
    unsafe { arc.assume_init() }
}
