// We want to test that granting a SharedReadWrite will be added
// *below* an already granted SharedReadWrite -- so writing to
// the SharedReadWrite will invalidate the SharedReadWrite.
//@normalize-stderr-test: "0x[0-9a-fA-F]+" -> "$$HEX"

use std::cell::RefCell;
use std::mem;

fn main() {
    unsafe {
        let x = &mut RefCell::new(0);
        let y: &i32 = mem::transmute(&*x.borrow()); // launder lifetime
        let shr_rw = &*x; // thanks to interior mutability this will be a SharedReadWrite
        shr_rw.replace(1);
        let _val = *y; //~ ERROR: /read access .* tag does not exist in the borrow stack/
    }
}
