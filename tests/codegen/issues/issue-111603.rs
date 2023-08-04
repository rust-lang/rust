// compile-flags: -O

// FIXME: DO NOT MERGE until these tests are re-enabled

#![crate_type = "lib"]
#![feature(get_mut_unchecked, new_uninit)]

use std::sync::Arc;

// CHECK-LABEL: @new_from_array
#[no_mangle]
pub fn new_from_array(x: u64) -> Arc<[u64]> {
    // Ensure that we only generate one alloca for the array.

    // CHECK: alloca
    // CHECK-SAME: [1000 x i64]
    // CHECK-NOT: alloca
    let array = [x; 1000];
    Arc::new(array)
}

// CHECK-LABEL: @new_uninit
#[no_mangle]
pub fn new_uninit(x: u64) -> Arc<[u64; 1000]> {
    let mut arc = Arc::new_uninit();
    unsafe { Arc::get_mut_unchecked(&mut arc) }.write([x; 1000]);
    unsafe { arc.assume_init() }
}

// CHECK-LABEL: @new_uninit_slice
#[no_mangle]
pub fn new_uninit_slice(x: u64) -> Arc<[u64]> {
    let mut arc = Arc::new_uninit_slice(1000);
    for elem in unsafe { Arc::get_mut_unchecked(&mut arc) } {
        elem.write(x);
    }
    unsafe { arc.assume_init() }
}
