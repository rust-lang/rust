//! Random data generation using the Zircon kernel.
//!
//! Fuchsia, as always, is quite nice and provides exactly the API we need:
//! <https://fuchsia.dev/reference/syscalls/cprng_draw>.

use crate::io::BorrowedCursor;

#[link(name = "zircon")]
unsafe extern "C" {
    fn zx_cprng_draw(buffer: *mut u8, len: usize);
}

pub fn fill_buf(&mut self, mut cursor: BorrowedCursor<'_, u8>) {
    unsafe {
        zx_cprng_draw(cursor.as_mut().as_mut_ptr().cast(), cursor.capacity());
        cursor.advance(cursor.capacity());
    }
}
