use core::ptr;

fn main() {
    unsafe {
        let base = ptr::without_provenance::<()>(10);
        let unit = &*base;
        let p1 = unit as *const ();

        let base = ptr::without_provenance::<()>(11);
        let unit = &*base;
        let p2 = unit as *const ();

        // Seems to work because they are same pointer
        // even though it's dangling.
        let _ = p1.byte_offset_from(p1);

        // UB because different pointers.
        let _ = p1.byte_offset_from(p2); //~ERROR: not both derived from the same allocation
    }
}
