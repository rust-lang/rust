// Make sure we catch this even without validation
//@compile-flags: -Zmiri-disable-validation

// Make sure that we cannot load from memory a `&mut` that got already invalidated.
fn main() {
    let x = &mut 42;
    let xraw = x as *mut _;
    let xref = unsafe { &mut *xraw };
    let xref_in_mem = Box::new(xref);
    let _val = unsafe { *xraw }; // invalidate xref
    let _val = *xref_in_mem; //~ ERROR: /retag .* tag does not exist in the borrow stack/
}
