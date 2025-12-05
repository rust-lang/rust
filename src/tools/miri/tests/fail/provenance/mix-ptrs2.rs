use std::mem;

const PTR_SIZE: usize = mem::size_of::<&i32>();

/// Overwrite one byte of a pointer, then restore it.
fn main() {
    unsafe fn ptr_bytes<'x>(ptr: &'x mut *const i32) -> &'x mut [mem::MaybeUninit<u8>; PTR_SIZE] {
        mem::transmute(ptr)
    }

    // Returns a value with the same provenance as `x` but 0 for the integer value.
    // `x` must be initialized.
    unsafe fn zero_with_provenance(x: mem::MaybeUninit<u8>) -> mem::MaybeUninit<u8> {
        let ptr = [x; PTR_SIZE];
        let ptr: *const i32 = mem::transmute(ptr);
        let mut ptr = ptr.with_addr(0);
        ptr_bytes(&mut ptr)[0]
    }

    unsafe {
        let ptr = &42;
        let mut ptr = ptr as *const i32;
        // Get a bytewise view of the pointer.
        let ptr_bytes = ptr_bytes(&mut ptr);

        // The highest bytes must be 0 for this to work.
        let hi = if cfg!(target_endian = "little") { ptr_bytes.len() - 1 } else { 0 };
        assert_eq!(*ptr_bytes[hi].as_ptr().cast::<u8>(), 0);
        // Overwrite provenance on the last byte.
        ptr_bytes[hi] = mem::MaybeUninit::new(0);
        // Restore it from the another byte.
        ptr_bytes[hi] = zero_with_provenance(ptr_bytes[1]);

        // Now ptr is almost good, except the provenance fragment indices do not work out...
        assert_eq!(*ptr, 42); //~ERROR: no provenance
    }
}
