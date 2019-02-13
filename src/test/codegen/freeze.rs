// compile-flags: -O
#![crate_type="lib"]
#![feature(ptr_freeze)]

use std::ptr;
use std::mem;

// freeze should prevent reads of uninitialized memory from being UB
#[no_mangle]
pub fn read_uninitialized() -> u8 {
    // CHECK-LABEL: @read_uninitialized
    // CHECK-NOT: undef
    unsafe {
        let mut v: u8 = mem::uninitialized();
        ptr::freeze(&mut v, 1);
        v
    }
}
