// A callee may not read the destination of our `&mut` without
// us noticing.

fn main() {
    let mut x = 15;
    let xraw = &mut x as *mut _;
    let xref = unsafe { &mut *xraw }; // derived from raw, so using raw is still ok...
    callee(xraw);
    let _val = *xref; // ...but any use of raw will invalidate our ref.
    //~^ ERROR: borrow stack
}

fn callee(xraw: *mut i32) {
    let _val = unsafe { *xraw };
}
