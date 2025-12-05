/// This tests that when a field sits at a well-aligned offset, accessing the field
/// requires high alignment even if the field type has lower alignment requirements.

#[repr(C, align(16))]
#[derive(Default, Copy, Clone)]
pub struct Aligned {
    _pad: [u8; 11],
    packed: Packed,
}
#[repr(C, packed)]
#[derive(Default, Copy, Clone)]
pub struct Packed {
    _pad: [u8; 5],
    x: u8,
}

unsafe fn foo(x: *const Aligned) -> u8 {
    unsafe { (*x).packed.x } //~ERROR: based on pointer with alignment 1, but alignment 16 is required
}

fn main() {
    unsafe {
        let mem = [Aligned::default(); 16];
        let odd_ptr = std::ptr::addr_of!(mem).cast::<u8>().add(1);
        // `odd_ptr` is now not aligned enough for `Aligned`.
        // If accessing the nested field `packed.x` can exploit that it is at offset 16
        // in a 16-aligned struct, this has to be UB.
        foo(odd_ptr.cast());
    }
}
