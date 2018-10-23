// Make sure that we cannot load from memory a `&mut` that got already invalidated.
fn main() {
    let x = &mut 42;
    let xraw = x as *mut _;
    let xref = unsafe { &mut *xraw };
    let xref_in_mem = Box::new(xref);
    let _val = *x; // invalidate xraw
    let _val = *xref_in_mem; //~ ERROR Mut reference with non-reactivatable tag Mut(Uniq
}
