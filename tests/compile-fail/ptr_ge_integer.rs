fn main() {
    let b = Box::new(0);
    let x = &*b as *const i32;
    // We cannot test if this is >= 64.  After all, depending on the base address, that
    // might or might not be the case.
    assert!(x >= 64 as *const i32); //~ ERROR invalid arithmetic on pointers
}
