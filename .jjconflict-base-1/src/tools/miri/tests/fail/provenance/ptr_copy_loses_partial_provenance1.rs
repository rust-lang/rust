fn main() {
    let half_ptr = std::mem::size_of::<*const ()>() / 2;
    let mut bytes = [1u8; 16];
    let bytes = bytes.as_mut_ptr();

    unsafe {
        // Put a pointer in the middle.
        bytes.add(half_ptr).cast::<&i32>().write_unaligned(&42);
        // Typed copy of the entire thing as two *function* pointers, but not perfectly
        // overlapping with the pointer we have in there.
        let copy = bytes.cast::<[fn(); 2]>().read_unaligned();
        let copy_bytes = copy.as_ptr().cast::<u8>();
        // Now go to the middle of the copy and get the pointer back out.
        let ptr = copy_bytes.add(half_ptr).cast::<*const i32>().read_unaligned();
        // Dereferencing this should fail as the copy has removed the provenance.
        let _val = *ptr; //~ERROR: dangling
    }
}
