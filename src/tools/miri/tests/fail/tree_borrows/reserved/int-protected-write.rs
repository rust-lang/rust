// We disable the GC for this test because it would change what is printed.
//@compile-flags: -Zmiri-tree-borrows -Zmiri-provenance-gc=0

#[path = "../../../utils/mod.rs"]
#[macro_use]
mod utils;

// Check how a Reserved without interior mutability responds to a Foreign
// Write when under a protector
fn main() {
    unsafe {
        let n = &mut 0u8;
        name!(n);
        let x = &mut *(n as *mut _);
        name!(x);
        let y = (&mut *n) as *mut _;
        name!(y);
        write_second(x, y);
        unsafe fn write_second(x: &mut u8, y: *mut u8) {
            let alloc_id = alloc_id!(x);
            name!(x, "callee:x");
            name!(x=>1, "caller:x");
            name!(y, "callee:y");
            name!(y, "caller:y");
            print_state!(alloc_id);
            // Right before the faulty Write, x is
            // - Reserved
            // - Protected
            // The Write turns it Disabled
            *y = 0; //~ ERROR: /write access through .* is forbidden/
        }
    }
}
