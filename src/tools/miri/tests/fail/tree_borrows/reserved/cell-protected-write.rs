// We disable the GC for this test because it would change what is printed.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-provenance-gc=0

// Check how a Reserved with interior mutability
// responds to a Foreign Write under a Protector
#[path = "../../../utils/mod.rs"]
#[macro_use]
mod utils;

use std::cell::UnsafeCell;

fn main() {
    unsafe {
        let n = &mut UnsafeCell::new(0u8);
        name!(n as *mut _, "base");
        let x = &mut *(n as *mut UnsafeCell<_>);
        name!(x as *mut _, "x");
        let y = (&mut *n) as *mut UnsafeCell<_> as *mut _;
        name!(y);
        write_second(x, y);
        unsafe fn write_second(x: &mut UnsafeCell<u8>, y: *mut u8) {
            let alloc_id = alloc_id!(x.get());
            name!(x as *mut _, "callee:x");
            name!((x as *mut _)=>1, "caller:x");
            name!(y, "callee:y");
            name!(y, "caller:y");
            print_state!(alloc_id);
            // Right before the faulty Write, x is
            // - Reserved
            // - with interior mut
            // - Protected
            *y = 1; //~ ERROR: /write access through .* is forbidden/
        }
    }
}
