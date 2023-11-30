#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::ptr;

#[repr(C)]
struct S(u8, u16);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let s: S;
        let sptr;
        let sptr2;
        let _val;
        {
            sptr = ptr::addr_of_mut!(s);
            sptr2 = sptr as *mut [u8; 4];
            *sptr2 = [0; 4];
            *sptr = S(0, 0); // should reset the padding
            _val = *sptr2; // should hence be UB
            //~^ERROR: encountered uninitialized memory
            Return()
        }
    }
}
