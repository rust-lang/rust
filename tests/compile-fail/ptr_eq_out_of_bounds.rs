fn main() {
    let b = Box::new(0);
    let x = (&*b as *const i32).wrapping_sub(0x800); // out-of-bounds
    let b = Box::new(0);
    let y = &*b as *const i32; // different allocation
    // We cannot compare these even though both allocations are live -- they *could* be
    // equal (with the right base addresses).
    assert!(x != y); //~ ERROR invalid arithmetic on pointers
}
