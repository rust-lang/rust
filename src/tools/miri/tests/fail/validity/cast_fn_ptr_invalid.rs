//! Even just *casting* a function pointer, withot ever calling it, requires validity.
#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]
#![allow(unused_assignments)]

use core::intrinsics::mir::*;

// Overwrites `ptr` by invoking the callback, then casts `ptr` to `*mut u8`.
// Needs to use custom MIR to avoid copies that perform their own validation.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn test(ptr: fn(), overwrite: fn(&mut fn())) {
    mir! {
        let ptrptr;
        let ptr2 : *mut u8;
        let _unused;

        {
            ptrptr = &mut ptr;
            Call(_unused = overwrite(ptrptr), ReturnTo(ret), UnwindContinue())
        }

        ret = {
            ptr2 = ptr as *mut u8; //~ERROR: does not point to a function
            Return()
        }
    }
}

fn f() {}

fn main() {
    test(f, |ptrptr| unsafe {
        let ptrptr = std::ptr::from_mut(ptrptr);
        ptrptr.cast::<*const ()>().write(&())
    });
}
