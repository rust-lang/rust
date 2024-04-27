/// This tests that when a field sits at offset 0 in a 4-aligned struct, accessing the field
/// requires alignment 4 even if the field type has lower alignment requirements.

#[repr(C)]
pub struct S {
    x: u8,
    y: u32,
}

unsafe fn foo(x: *const S) -> u8 {
    unsafe { (*x).x } //~ERROR: based on pointer with alignment 1, but alignment 4 is required
}

fn main() {
    unsafe {
        let mem = [0u64; 16];
        let odd_ptr = std::ptr::addr_of!(mem).cast::<u8>().add(1);
        // `odd_ptr` is now not aligned enough for `S`.
        // If accessing field `x` can exploit that it is at offset 0
        // in a 4-aligned struct, that field access requires alignment 4,
        // thus making this UB.
        foo(odd_ptr.cast());
    }
}
