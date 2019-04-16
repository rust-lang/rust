// We *can* have aliasing &RefCell<T> and &mut T, but we cannot read through the former.
// Else we couldn't optimize based on the assumption that `xref` below is truly unique.

use std::cell::RefCell;
use std::{mem, ptr};

fn main() {
    let rc = RefCell::new(0);
    let mut refmut = rc.borrow_mut();
    let xref: &mut i32 = &mut *refmut;
    let xshr = &rc; // creating this is ok
    let _val = *xref; // we can even still use our mutable reference
    mem::forget(unsafe { ptr::read(xshr) }); // but after reading through the shared ref
    let _val = *xref; // the mutable one is dead and gone
    //~^ ERROR borrow stack
}
