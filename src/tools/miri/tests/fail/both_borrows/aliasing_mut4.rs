//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[tree]error-in-other-file: /write access through .* is forbidden/
use std::cell::Cell;
use std::mem;

// Make sure &mut UnsafeCell also is exclusive
pub fn safe(x: &i32, y: &mut Cell<i32>) {
    //~[stack]^ ERROR: protect
    y.set(1);
    let _ = *x;
}

fn main() {
    let mut x = 0;
    let xref = &mut x;
    let xraw: *mut i32 = unsafe { mem::transmute_copy(&xref) };
    let xshr = &*xref;
    // transmute fn ptr around so that we can avoid retagging
    let safe_raw: fn(x: *const i32, y: *mut Cell<i32>) =
        unsafe { mem::transmute::<fn(&i32, &mut Cell<i32>), _>(safe) };
    safe_raw(xshr, xraw as *mut _);
}
