use std::cell::UnsafeCell;

fn main() { unsafe {
    let c = &UnsafeCell::new(UnsafeCell::new(0));
    let inner_uniq = &mut *c.get();
    let inner_shr = &*inner_uniq; // a SharedRW with a tag
    *c.get() = UnsafeCell::new(1); // invalidates the SharedRW
    let _val = *inner_shr.get(); //~ ERROR borrow stack
    let _val = *inner_uniq.get();
} }
