fn main() {
    let target = Box::new(42); // has an implicit raw
    let xref = &*target;
    {
        let x: *mut u32 = xref as *const _ as *mut _;
        unsafe { *x = 42 }; //~ ERROR: /write access .* tag only grants SharedReadOnly permission/
    }
    let _x = *xref;
}
