#![allow(integer_to_ptr_transmutes)]

fn main() {
    #[cfg(all(target_endian = "little", target_pointer_width = "64"))]
    let bad = unsafe { std::mem::transmute::<u128, &[u8]>(42) };
    #[cfg(all(target_endian = "big", target_pointer_width = "64"))]
    let bad = unsafe { std::mem::transmute::<u128, &[u8]>(42 << 64) };
    #[cfg(all(target_endian = "little", target_pointer_width = "32"))]
    let bad = unsafe { std::mem::transmute::<u64, &[u8]>(42) };
    #[cfg(all(target_endian = "big", target_pointer_width = "32"))]
    let bad = unsafe { std::mem::transmute::<u64, &[u8]>(42 << 32) };
    // This created a slice with length 0, so the following will fail the bounds check.
    bad[0];
}
