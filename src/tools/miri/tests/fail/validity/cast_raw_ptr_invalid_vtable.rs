#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]
#![allow(unused_assignments)]

use core::intrinsics::mir::*;

// Overwrites `ptr` by invoking the callback, then casts `ptr` to `*mut u8`.
// Needs to use custom MIR to avoid copies that perform their own validation.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn test<T: ?Sized, U>(ptr: *const T, data: U, overwrite: fn(&mut *const T, U)) {
    mir! {
        let ptrptr;
        let ptr2 : *mut u8; // cast drops metadata!
        let _unused;

        {
            ptrptr = &mut ptr;
            Call(_unused = overwrite(ptrptr, data), ReturnTo(ret), UnwindContinue())
        }

        ret = {
            ptr2 = CastPtrToPtr(ptr); //~ERROR: vtable for `std::fmt::Debug` but `std::fmt::Display` was expected
            Return()
        }
    }
}

#[allow(unused)]
struct S<Tail: ?Sized> {
    f: i32,
    g: Tail,
}

fn main() {
    let x = S { f: 0, g: 0 };
    let ptr1: *const S<dyn std::fmt::Debug> = &x;
    let ptr2: *const S<dyn std::fmt::Display> = &x;
    test::<S<dyn std::fmt::Display>, _>(ptr2, ptr1, |ptrptr2, ptr1| unsafe {
        // Give ptr2 the vtable from ptr1.
        let ptrptr2 = std::ptr::from_mut(ptrptr2);
        ptrptr2.copy_from(&raw const ptr1 as *const _, 1);
    });
}
