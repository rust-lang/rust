//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// Create zero-sized references to vtables and function data.
// Just make sure nothing explodes.

use std::{mem, ptr};

fn check_ref(x: &()) {
    let _ptr = ptr::addr_of!(*x);
}

fn main() {
    check_ref({
        // Create reference to a function.
        let fnptr: fn(&()) = check_ref;
        unsafe { mem::transmute(fnptr) }
    });
    check_ref({
        // Create reference to a vtable.
        let wideptr: &dyn Send = &0;
        let fields: (&i32, &()) = unsafe { mem::transmute(wideptr) };
        fields.1
    })
}
