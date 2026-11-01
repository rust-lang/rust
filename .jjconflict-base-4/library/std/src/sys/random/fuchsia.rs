//! Random data generation using the Zircon kernel.
//!
//! Fuchsia, as always, is quite nice and provides exactly the API we need:
//! <https://fuchsia.dev/reference/syscalls/cprng_draw>.

#[link(name = "zircon")]
unsafe extern "C" {
    fn zx_cprng_draw(buffer: *mut u8, len: usize);
}

pub fn fill_bytes(bytes: &mut [u8]) {
    unsafe { zx_cprng_draw(bytes.as_mut_ptr(), bytes.len()) }
}
