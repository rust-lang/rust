//@compile-flags: -Zmiri-tree-borrows

use core::cell::UnsafeCell;
use core::mem;

fn main() {
    unsafe {
        let x = &0i32;
        // As long as we only read, transmuting this to UnsafeCell should be fine.
        let cell_x: &UnsafeCell<i32> = mem::transmute(&x);
        let _val = *cell_x.get();
    }
}
