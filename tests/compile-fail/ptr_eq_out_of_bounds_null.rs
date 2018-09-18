fn main() {
    let b = Box::new(0);
    let x = (&*b as *const i32).wrapping_sub(0x800); // out-of-bounds
    // We cannot compare this with NULL. After all, this *could* be NULL (with the right base address).
    assert!(x != std::ptr::null()); //~ ERROR invalid arithmetic on pointers
}
