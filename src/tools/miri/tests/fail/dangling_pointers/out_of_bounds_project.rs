// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation
use std::ptr::addr_of;

fn main() {
    let v = 0u32;
    let ptr = addr_of!(v).cast::<(u32, u32, u32)>();
    unsafe {
        let _field = addr_of!((*ptr).1); // still just in-bounds
        let _field = addr_of!((*ptr).2); //~ ERROR: in-bounds pointer arithmetic failed
    }
}
