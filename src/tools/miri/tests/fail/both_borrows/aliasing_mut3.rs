//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::mem;

fn safe(x: &mut i32, y: &i32) {
    //~[stack]^ ERROR: borrow stack
    //~[tree]^^ ERROR: /Undefined Behavior: reborrow through .* at .* is forbidden/
    *x = 1;
    let _v = *y;
}

fn main() {
    let mut x = 0;
    let xref = &mut x;
    let xraw: *mut i32 = unsafe { mem::transmute_copy(&xref) };
    let xshr = &*xref;
    // transmute fn ptr around so that we can avoid retagging
    let safe_raw: fn(x: *mut i32, y: *const i32) =
        unsafe { mem::transmute::<fn(&mut i32, &i32), _>(safe) };
    safe_raw(xraw, xshr);
}
