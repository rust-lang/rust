fn main() {
    let target = Box::new(42); // has an implicit raw
    let xref = &*target;
    {
        let x : *mut u32 = xref as *const _ as *mut _;
        unsafe { *x = 42; } // invalidates shared ref, activates raw
    }
    let _x = *xref; //~ ERROR borrow stack
}
